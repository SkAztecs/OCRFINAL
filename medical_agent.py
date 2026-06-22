import os
import sys
import gc
import glob
import io
import json
import json_repair #type: ignore
import logging
import warnings
import time
import multiprocessing
import re
from contextlib import redirect_stdout
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import torch # type: ignore 
import torch.utils._pytree as pytree # type: ignore
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig # type: ignore
from dotenv import load_dotenv
from tqdm import tqdm
import traceback
# from concurrent.futures import ThreadPoolExecutor  # REMOVED TO PREVENT RAM CRASH
from pydantic import BaseModel, Field
from datetime import datetime
from dateutil.relativedelta import relativedelta
import cv2
import numpy as np

try:
    import pypdfium2 as pdfium
except ImportError:
    raise ImportError("pypdfium2 is required. Install it: pip install pypdfium2")

try:
    from paddleocr import PaddleOCR  # type: ignore
    _PADDLE_AVAILABLE = True
    # PaddleOCR = None <--- REMOVED: This was breaking PaddleOCR!
except ImportError:
    _PADDLE_AVAILABLE = False

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# ============================================================================
# 1. ENVIRONMENT & CONFIGURATION
# ============================================================================
load_dotenv()
 
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]            = "false"
 
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
 
PDF_PATH          = "final1.pdf"
RENDER_SCALE      = 2.5
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR        = os.path.join(BASE_DIR, "ocr_output")
OLLAMA_OCR_MODEL  = "maternion/LightOnOCR-2:1b"
COMPRESSION_MODE  = "base"
OLLAMA_MODEL      = "qwen3:8b"         
OLLAMA_BASE_URL   = "http://localhost:11434"  # Default Ollama server address
OLLAMA_OPTIONS    = {
    "temperature": 0,
    "num_predict": 10240,
    "num_ctx": 24576,
    "think": False,
}

ENHANCE_CONTRAST  = False
DOCUMENT_TYPE     = "auto"
FALSE_IMAGE_CHAR_THRESHOLD = 100

def _unload_model(model_name: str) -> None:
    """
    Force-evict a model from Ollama VRAM immediately, regardless of its
    keep_alive setting. Used to guarantee a clean GPU handoff between the
    OCR phase (LightOnOCR) and the extraction phase (qwen3:8b), since
    qwen3 is kept loaded with keep_alive=-1 across a job and would
    otherwise stay resident in VRAM during the next job's OCR phase,
    causing severe GPU contention/slowdown.
    """
    import urllib.request
    import json as _json
    try:
        payload = _json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        logger.info(f"  Unloaded {model_name} from VRAM.")
    except Exception as e:
        logger.warning(f"  Failed to unload {model_name}: {e}")

# ============================================================================
# OCR ENGINE SELECTION
# ============================================================================
SELECTABLE_TEXT_RATIO = 0.6   
MIN_CHARS_PER_PAGE    = 80    
PADDLE_USE_GPU        = False  
OCR_ENGINE            = "auto"  

_LIGHTON_ALIASES = frozenset({
    "deepseek", "lighton", "lightonocr",
})

def normalize_ocr_engine(engine: str) -> str:
    if not engine:
        return "auto"
    key = engine.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if key == "auto":
        return "auto"
    if key in ("paddleocr", "paddle"):
        return "paddleocr"
    if key in _LIGHTON_ALIASES:
        return "lighton"
    return engine.strip().lower()

def ocr_engine_label(engine: str) -> str:
    canonical = normalize_ocr_engine(engine)
    return {"auto": "AUTO", "paddleocr": "PaddleOCR", "lighton": "LightOnOCR"}.get(
        canonical, canonical.upper()
    )

PDF_BASENAME      = os.path.splitext(os.path.basename(PDF_PATH))[0]
PDF_WORKSPACE_DIR = os.path.join(OUTPUT_DIR, PDF_BASENAME)
os.makedirs(PDF_WORKSPACE_DIR, exist_ok=True)

STITCHED_PATH = os.path.join(OUTPUT_DIR, f"{PDF_BASENAME}_stitched.txt")
RESULT_PATH   = os.path.join(OUTPUT_DIR, f"{PDF_BASENAME}.json")
 
# ============================================================================
# 2. LOGGING CONFIGURATION
# ============================================================================
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
 
def get_configured_logger(name: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, "pipeline.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(name)
 
logger = get_configured_logger(__name__)
 
# ============================================================================
# 3. UTILITIES & COMPRESSION RESOLVER
# ============================================================================
_COMPRESSION_PRESETS = {
    "tiny":   dict(base_size=512,  image_size=512,  crop_mode=False),
    "small":  dict(base_size=640,  image_size=640,  crop_mode=False),
    "base":   dict(base_size=1024, image_size=1024, crop_mode=False),
    "large":  dict(base_size=1280, image_size=1280, crop_mode=False),
    "high":   dict(base_size=1024, image_size=640,  crop_mode=True),
}
 
def get_compression_params(mode: str) -> dict:
    mode = mode.lower().strip()
    if mode not in _COMPRESSION_PRESETS:
        mode = "small"
    params = _COMPRESSION_PRESETS[mode]
    return params
 
def free_memory(model=None, tokenizer=None) -> None:
    if model is not None:
        try: del model
        except Exception: pass
    if tokenizer is not None:
        try: del tokenizer
        except Exception: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logger.info("  Memory freed.")
 
def log_vram(label: str = "") -> None:
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"  VRAM [{label}]: {used:.2f} GB / {total:.2f} GB")
 
# ============================================================================
# 4. FALSE-IMAGE DETECTION HELPERS
# ============================================================================
_BBOX_PATTERN    = re.compile(r'\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]')
_TAG_PATTERN     = re.compile(r'<\|(?:ref|det|/ref|/det)\|>')
_FULL_PAGE_BBOX  = re.compile(r'\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]')
 
def _strip_ocr_artifacts(text: str) -> str:
    text = _TAG_PATTERN.sub('', text)
    text = _BBOX_PATTERN.sub('', text)
    return text.strip()
 
def _is_false_image(text: str) -> bool:
    clean = _strip_ocr_artifacts(text)
    if len(clean) < FALSE_IMAGE_CHAR_THRESHOLD:
        return True
    boxes = _FULL_PAGE_BBOX.findall(text)
    if len(boxes) == 1:
        x1, y1, x2, y2 = (int(v) for v in boxes[0])
        width_frac  = (x2 - x1) / 999
        height_frac = (y2 - y1) / 999
        if width_frac >= 0.85 and height_frac >= 0.85:
            return True
    return False

def deskew(gray: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 50:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def suppress_glare(gray: np.ndarray) -> np.ndarray:
    _, glare_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    glare_ratio = glare_mask.sum() / (255 * gray.size)
    if 0.001 < glare_ratio < 0.15:
        gray = cv2.inpaint(gray, glare_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return gray

def adaptive_denoise(gray: np.ndarray) -> np.ndarray:
    noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()
    if noise_level > 500:
        h = 18
    elif noise_level > 150:
        h = 10
    else:
        h = 4
    return cv2.fastNlMeansDenoising(gray, h=h, templateWindowSize=7, searchWindowSize=21)

def unsharp_mask(img: np.ndarray, strength: float = 0.6) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)

def is_already_clean(gray: np.ndarray) -> bool:
    std = gray.std()
    dark_ratio = (gray < 50).sum() / gray.size
    return std > 60 and dark_ratio < 0.02

def enhance_document_cv2(pil_img: Image.Image) -> Image.Image:
    img_cv = np.array(pil_img)
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_cv

    if is_already_clean(gray):
        res_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(res_rgb)

    h, w = gray.shape[:2]
    if min(h, w) < 1200:
        scale = 1200 / min(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = deskew(gray)
    gray = suppress_glare(gray)
    gray = adaptive_denoise(gray)

    h, w = gray.shape[:2]
    k = max(21, (min(h, w) // 12) | 1)

    bg_img = cv2.GaussianBlur(gray, (k, k), 0)
    flat_img = cv2.divide(gray, bg_img, scale=255)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    final_img = unsharp_mask(clahe.apply(flat_img))

    res_rgb = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(res_rgb)
 
# ============================================================================
# 8. OCR PIPELINE FUNCTIONS
# ============================================================================
def pdf_to_images(pdf_path: str, scale: float = 2.0, enhance_contrast: bool = False) -> int:
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return 0
 
    try:
        pdf   = pdfium.PdfDocument(pdf_path)
        total = len(pdf)
        logger.info(f"  PDF has {total} page(s). Rendering to workspace: {PDF_WORKSPACE_DIR}")
        for i in range(total):
            page_num = i + 1
            out_path = os.path.join(PDF_WORKSPACE_DIR, f"page_{page_num}.png")
            
            bitmap   = pdf[i].render(scale=scale)
            img      = bitmap.to_pil().convert("RGB")
            
            if enhance_contrast:
                img = enhance_document_cv2(img)
                
            img.save(out_path)
            
            del img
            del bitmap
            
            if page_num % 10 == 0 or page_num == total:
                logger.info(f"    Rendered page {page_num}/{total}")
                
        pdf.close()
        return total
    except Exception as e:
        logger.error(f"pypdfium2 render failed: {e}")
        return 0
 
def load_ocr_model():
    logger.info("  LightOnOCR via Ollama — skipping local model load.")
    return None, None 
 
def transcribe_all_pages(
    total_pages: int,
    model,          
    tokenizer,      
    compression_mode: str = "small",
    force_rerun: bool = False,
) -> List[Tuple[int, str]]:
    import base64
    import urllib.request

    page_texts: List[Tuple[int, str]] = []
    prompt = (
        "You are an OCR engine. Extract ALL text from this medical document image "
        "exactly as printed. Preserve every word, number, date, unit, table cell, "
        "and symbol. Output plain text with no commentary."
    )

    # Ensure qwen3:8b (kept loaded with keep_alive=-1 by the extraction phase
    # of a prior job) is evicted before OCR starts, so LightOnOCR doesn't run
    # concurrently with qwen3 resident in VRAM. Without this, every job after
    # the first one runs OCR while qwen3 sits in VRAM "Forever", causing the
    # ~9-10s/page -> ~44s/page slowdown.
    _unload_model(OLLAMA_MODEL)

    pbar = tqdm(range(1, total_pages + 1), desc="OCR pages", unit="page", dynamic_ncols=True)
    for page_num in pbar:
        pbar.set_postfix(page=f"{page_num}/{total_pages}", status="starting")
        img_path = os.path.join(PDF_WORKSPACE_DIR, f"page_{page_num}.png")
        out_path = os.path.join(PDF_WORKSPACE_DIR, f"raw_text_page_{page_num}.txt")

        if not force_rerun and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                text = f.read()
            if text.strip():
                page_texts.append((page_num, text))
                pbar.set_postfix(page=f"{page_num}/{total_pages}", status="cached ✓")
                continue

        if not os.path.exists(img_path):
            logger.error(f"  Image missing: {img_path}")
            page_texts.append((page_num, ""))
            continue

        try:
            from PIL import Image
            import io
            
            # Open the massive high-res image
            with Image.open(img_path) as img:
                # Shrink it to a max dimension of 1024px (maintaining aspect ratio)
                # This drastically reduces the token count for LightOnOCR
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                # Convert the shrunken image to Base64 in memory
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            payload = json.dumps({
                "model":  OLLAMA_OCR_MODEL,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "keep_alive": "5m",  # <--- CRITICAL FIX: Instantly unloads Vision model from RAM
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # Increased timeout to 900 to prevent premature cutoffs
            with urllib.request.urlopen(req, timeout=900) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            text = result.get("response", "").strip()
            logger.info(f"  Page {page_num}: {len(text)} chars.")
        except Exception as e:
            logger.error(f"  Page {page_num} OCR failed: {e}")
            text = ""

        if text.strip():
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            page_texts.append((page_num, text))
            pbar.set_postfix(page=f"{page_num}/{total_pages}", status=f"done ✓ ({len(text)} chars)")
        else:
            logger.warning(f"  Page {page_num} returned empty text. Not caching.")
            pbar.set_postfix(page=f"{page_num}/{total_pages}", status="failed ✗")

    pbar.close()
    return page_texts
 
def stitch_pages(page_texts: List[Tuple[int, str]], stitched_path: str = None) -> str:
    _stitched_path = stitched_path or STITCHED_PATH
    valid = [(n, t.strip()) for n, t in page_texts if t and t.strip()]
    if not valid:
        logger.warning("  No text extracted from any page.")
        return ""
        
    combined = "\n\n".join(f"--- PAGE {n} ---\n{t}" for n, t in valid)
 
    with open(_stitched_path, "w", encoding="utf-8") as f:
        f.write(combined)
    logger.info(f"  Stitched {len(valid)} page(s) → {len(combined)} chars total -> {_stitched_path}")
    return combined

# ============================================================================
# 8.5 STRUCTURED OUTPUT SCHEMAS & POST-PROCESSING
# ============================================================================
class Address(BaseModel):
    addressLine1: Optional[str] = None
    addressLine2: Optional[str] = None
    ward: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    pincode: Optional[str] = None

class PersonalInfo(BaseModel):
    patientName: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    dob: Optional[str] = None
    phoneNo: Optional[str] = None
    rawAddress: Optional[str] = None
    address: Optional[Address] = None

class MedicalEvent(BaseModel):
    date: str = Field(description="Date in DD MMM, YYYY format")
    category: str = Field(description="e.g., radiologyExamination, surgery, medications, labResult, clinicalExamination")
    content: Any = Field(description="The full, unsummarized extracted details for this event.")

class DocumentData(BaseModel):
    documentDate: Optional[str] = None
    hospitalName: Optional[str] = None
    documentName: Optional[str] = None
    hospitalId: Optional[str] = None
    patientId: Optional[str] = None
    personalInfos: Optional[PersonalInfo] = None
    events: Optional[List[MedicalEvent]] = None
    generalNotes: Optional[str] = Field(None, description="Extract any clinical summaries.")

class ExtractionResponse(BaseModel):
    documents: List[DocumentData]
    parsedDocumentPercentage: int = Field(description="Set to 100 if extraction succeeded completely.")

def calculate_dob(age_string: str, doc_date_string: str) -> str:
    if not age_string or not doc_date_string:
        return None
    try:
        if len(doc_date_string) == 4:
            doc_date = datetime(int(doc_date_string), 1, 1)
        else:
            doc_date = datetime.strptime(doc_date_string, "%d %b, %Y")
            
        years_match = re.search(r'(\d+)\s*(y|yr|yrs|year|years)', age_string, re.I)
        months_match = re.search(r'(\d+)\s*(mth|mths|mnth|mo|mos|month|months)', age_string, re.I)
        days_match = re.search(r'(\d+)\s*(d|dy|dys|day|days)', age_string, re.I)

        if not years_match and re.search(r'^(\d+)(/|$)', age_string):
            years = int(re.search(r'^(\d+)', age_string).group(1))
        else:
            years = int(years_match.group(1)) if years_match else 0
            
        months = int(months_match.group(1)) if months_match else 0
        days = int(days_match.group(1)) if days_match else 0
        
        dob = doc_date - relativedelta(years=years, months=months, days=days)
        return dob.strftime("%Y-%m-%d")
    except Exception:
        return None

def parse_address(raw_address: str) -> dict:
    if not raw_address:
        return None
    address_data = {"addressLine1": None, "addressLine2": None, "ward": None, "city": None, "district": None, "state": None, "country": None, "pincode": None}
    pincode_match = re.search(r'\b(\d{6})\b', raw_address)
    address_data["pincode"] = pincode_match.group(1) if pincode_match else None
    ward_match = re.search(r'(?i)\b(ward\s*(?:no\.?|number)?\s*[\w-]+)\b', raw_address)
    address_data["ward"] = ward_match.group(1).strip() if ward_match else None
    parts = [p.strip() for p in raw_address.split(',') if p.strip()]
    if len(parts) >= 2:
        address_data["addressLine1"] = parts[0]
        address_data["addressLine2"] = parts[1] if len(parts) > 2 else None
        address_data["city"] = parts[-2] if not pincode_match else parts[-1].replace(address_data["pincode"], "").strip()
    else:
        address_data["addressLine1"] = raw_address
    return address_data

def apply_post_processing(extracted_dict: dict) -> dict:
    if not isinstance(extracted_dict, dict) or "documents" not in extracted_dict:
        return extracted_dict
    for doc in extracted_dict["documents"]:
        doc_date = doc.get("documentDate")
        personal_info = doc.get("personalInfos", {})
        if personal_info:
            raw_age = personal_info.get("age")
            raw_address = personal_info.get("rawAddress")
            personal_info["dob"] = calculate_dob(raw_age, doc_date)
            if raw_address:
                personal_info["address"] = parse_address(raw_address)
                del personal_info["rawAddress"] 
    return extracted_dict

# ============================================================================
# 9. LLM PROMPTS & EXTRACTION LOGIC
# ============================================================================
CLASSIFIER_PROMPT = """You are a medical document classifier.
Read the document below. It is divided by --- PAGE X --- markers.
Identify all distinct medical documents present and the specific pages they occupy.

Return ONLY a JSON array of objects with this exact structure:
[
  {{
    "document_type": "discharge_summary",
    "start_page": 1,
    "end_page": 3
  }}
]

The "document_type" field MUST be EXACTLY one of the following string values — no other values are allowed:
  "discharge_summary"       — Inpatient discharge summary with admission/discharge dates
  "chemotherapy_admission"  — Chemo day-care / inpatient cycle record, pre-medications, infusion drugs
  "radiotherapy_report"     — Radiation treatment plan or session record (fractions, Gy, DVH)
  "pet_ct_scan"             — PET, PET-CT, or PET-MRI report (SUV values present)
  "ct_scan"                 — CT / CECT / NCCT / HRCT report (no PET component)
  "mri_scan"                — MRI report (T1, T2, FLAIR, DWI sequences)
  "mammogram"               — Mammography / tomosynthesis report (BI-RADS score)
  "histopathology_report"   — Histopathology, biopsy, HPE, or trucut report
  "cytology_report"         — FNAC or cytology report (smear / aspiration, no tissue block)
  "ultrasound_scan"         — Ultrasound / USG / sonography report
  "dna_test"                — Genetic / NGS / mutation panel / DNA test report
  "outpatient_note"         — OPD consultation note, clinic visit note, follow-up note
  "referral_letter"         — Referral letter from one doctor to another
  "registration_receipt"    — Hospital bill, receipt, or registration document
  "other"                   — Use ONLY when none of the above types fit

IMPORTANT: Copy the value EXACTLY as shown above (lowercase, underscores, no spaces).
Never invent new type names. If uncertain, choose "other".

DOCUMENT:
{combined_text}
"""

SYSTEM_PROMPT_DISCHARGE = """<system_role>
You are an expert clinical data extraction pipeline optimized for Oncology Electronic Health Records (EHR). Your primary directive is 100% data preservation and strict structural adherence. Missing a single drug, dose, TNM prefix, or molecular biomarker compromises patient safety.
</system_role>

<critical_directives>
0. DOCUMENT TYPE AUTO-DETECTION (DO THIS FIRST): Read the entire document and set `documentMetadata.documentType` to EXACTLY one of these values — no other values permitted:
   "Discharge Summary" | "Pathology/Biopsy Report" | "Radiology/Imaging Report" |
   "Laboratory/Blood Work" | "Operative/Surgery Note" | "Clinical/Progress Note" |
   "Prescription/Pharmacy" | "Other"
   For this pipeline, the document is almost always a "Discharge Summary". Only deviate if the content clearly does not match (e.g., the text is a biopsy report with no admission/discharge dates).
1. STRICT SCHEMA ADHERENCE: You must return ONLY a valid JSON object matching the <output_schema> exactly. Do not output conversational prose, markdown formatting outside of the JSON block, or preambles.2. NO MISSING KEYS: If a data field defined in the schema does not exist in the source text, you MUST populate it as `null` (for strings/objects) or `[]` (for arrays). NEVER modify, rename, or delete keys from the schema.
3. VERBATIM PRESERVATION: Never summarize imaging findings, pathology results, or clinical histories. Extract them exactly as written.
4. OCR RESILIENCE: Fix obvious OCR spacing errors (e.g., "BREASTINVASIVE" -> "BREAST, INVASIVE"), but NEVER alter numbers, percentages, measurements, or dates. Be vigilant for OCR errors in oncology markers (e.g., reading "Ki-67" as "K1-67", or "HER2" as "HE R2").
5. STRIP METADATA: Ignore all `<|ref|>`, `<|det|>` tags, and bounding box coordinates like `[[693, 342, 925, 353]]`.
6. ZERO TRUNCATION: For cancer patients, the historyOfPresentIllnessVerbatim, courseInHospitalVerbatim, all findingsVerbatim fields, and all grossDescription/microscopicDescription fields must be copied in full regardless of length. Do not end with "..." or summarize.
7. FINAL ONCOLOGY SAFETY VALIDATION:
- Before generating JSON:
  1. Count all medications in source.
  2. Count all imaging studies in source.
  3. Count all pathology studies in source.
  4. Count all biomarker entries in source.
  5. Count all follow-up instructions in source.
- Verify the counts match the JSON output.
- If any source item is not represented in JSON, regenerate extraction before producing output.
- No drug, biomarker, pathology result, imaging report, follow-up instruction, emergency warning symptom, or cancer staging element may be omitted.
</critical_directives>

<oncology_extraction_rules>
SECTION 1: DOCUMENT CLASSIFICATION & METADATA
- AUTO-DETECT DOCUMENT TYPE: Analyze the entire text and classify the document type into exactly ONE of the following schema ENUM options: ["Discharge Summary", "Pathology/Biopsy Report", "Radiology/Imaging Report", "Laboratory/Blood Work", "Operative/Surgery Note", "Clinical/Progress Note", "Prescription/Pharmacy", "Other"].
- Extract patient name, exact age string, gender, UHID/MRN, and full consultant names with credentials.
- Parse the address explicitly into line1, city, state, and pincode.

SECTION 2: CLINICAL NARRATIVE & CHIEF COMPLAINTS
- Extract chiefComplaints as a list of individual symptoms, not a single string.
- Extract historyOfPresentIllnessVerbatim completely and verbatim — do NOT truncate even if very long.
- Extract pastMedicalHistory, familyHistory, surgicalHistory if present.
- Extract smoking/alcohol/occupational exposure under socialHistory.

SECTION 3: ONCOLOGY STAGING & BIOMARKERS (CRITICAL)
- STAGING: Extract TNM staging with absolute fidelity to prefixes (`c`, `p`, `y`, `yp`). Do not strip these contextual letters.
- IHC & GENOMICS: Look actively for Next-Generation Sequencing (NGS) results, FISH, or molecular panels. Create explicit entries for ER, PR, HER2, BRCA, PD-L1, EGFR, ALK, ROS1, or Ki-67. Capture the marker, status, percentage, and clone.
CRITICAL BIOMARKER PRESERVATION:
- NEVER omit any IHC result.
- Extract every biomarker even if embedded inside a paragraph.
- Preserve:
  - ER
  - PR
  - HER2
  - Ki-67
  - PD-L1
  - BRCA1
  - BRCA2
  - EGFR
  - ALK
  - ROS1
  - MSI
  - TMB
  - NTRK
  - RET
  - MET
  - BRAF

- For each biomarker capture:
  biomarker
  result
  percentage
  score
  clone
  testingMethod
  specimenSite
  testDate

Example:
"HER2 (ERBB2) by immunohistochemistry (clone 4B5-Ventana) positive (score 3+)"
must generate:
{
  "biomarker":"HER2",
  "result":"Positive",
  "score":"3+",
  "clone":"4B5-Ventana",
  "testingMethod":"IHC"
}
LYMPH NODE RULE:
- Search separately for:
  axillary,
  cervical,
  supraclavicular,
  mediastinal,
  hilar,
  pelvic nodes.
- FNAC/HPE/Biopsy results from lymph nodes MUST NOT be merged with primary tumor pathology.
- Every nodal pathology finding must create a separate pathologyBiopsies entry.
SECTION 4: VERBATIM IMAGING & PATHOLOGY
- Build an array for every diagnostic scan (USG, Mammogram, PETCT, CT, MRI). Extract findings completely verbatim, including all anatomical measurements and BI-RADS metrics.
IMAGING COMPLETENESS:
- Do not omit any imaging modality.
- Capture:
  USG,
  Mammogram,
  PETCT,
  CT,
  MRI,
  Echo,
  X-ray,
  Bone Scan,
  PET-MRI.
- Preserve all measurements, SUV values, BI-RADS scores, node descriptions, and metastatic findings verbatim.
- Identify all histological biopsies (FNAC, Trucut, HPE). Capture specimen type, grade, and unique morphological hallmarks (e.g., "lymphatic emboli").

SECTION 5: LABORATORY WORKUPS
- Parse all inline or tabular laboratory panels (CBC, LFT, KFT, Cardiac) into clean test/value/status structures. Include LVEF and ECG findings.
CARDIAC MONITORING:
- Extract:
  LVEF,
  Echo findings,
  ECG findings,
  QT/QTc,
  cardiology opinions.
- Search entire document even if these appear outside laboratory sections.
SECTION 6: TREATMENT & TOXICITIES
- Map out the clinical protocol regimen (e.g., "TCHP") alongside active cycle numbers.
ONCOLOGY REGIMEN NORMALIZATION:
- Preserve both:
  protocol abbreviation,
  expanded drug list.

Example:
TCHP ->
Docetaxel
Carboplatin
Trastuzumab
Pertuzumab

Do not replace protocol abbreviations with expanded names.
Store both.
- TOXICITY: If the text mentions an "Adverse Event" or "Toxicity", extract the grading (e.g., "Grade 3") and note if it resulted in a dose reduction or delay.
- TUMOR RESPONSE: Scan for RECIST criteria outcomes (CR, PR, SD, PD).
TREATMENT DECISION PATHWAYS:
- Preserve conditional treatment decisions.
- Extract verbatim statements involving:
  PCR,
  pCR,
  RCB,
  RECIST,
  residual disease,
  escalation/de-escalation therapy,
  maintenance therapy.
- Store under treatmentExecution.futureTreatmentStrategy.

SECTION 7: DISCHARGE PHARMACOLOGY
- Scan carefully for all medication vectors: "Inj.", "Tab.", "Cap.", and "Syr.".
- Every distinct drug line must produce an entry mapping out name, dose, route, frequency (e.g., "1-1-1"), duration, and timing constraints (e.g., "before food"). Include pre-medications for the next cycle.
MEDICATION COMPLETENESS RULE:
- Every medication line after
  "ADVICE ON DISCHARGE"
  must create exactly one medicationsPrescribed entry.
- Do not merge medications.
- Preserve:
  name,
  strength,
  route,
  frequency,
  duration,
  timing,
  indication.
- Scan:
  Inj.
  Tab.
  Cap.
  Syr.
  Syp.
  Cream.
  Oint.
  Drops.
  Neb.
- Extract step-by-step follow-up dates and emergency warning symptoms.
FOLLOW-UP COMPLETENESS:
- Every follow-up instruction must generate a followUpSchedule entry.
- Do not merge instructions occurring on the same date.
- Preserve:
  date,
  action,
  responsible consultant,
  required investigations,
  pre-medications,
  document collection instructions.
  EMERGENCY ALERT RULE:
- Extract every warning sign listed under:
  Emergency,
  Red Flag Symptoms,
  Call Doctor If,
  Warning Symptoms.
- Preserve each symptom as a separate array element.
GENERAL EXAMINATION PRESERVATION:
- Extract all examination findings even if normal.
- Capture:
  Pulse,
  BP,
  Temperature,
  SpO2,
  Respiratory Rate,
  CVS,
  RS,
  Abdomen,
  CNS,
  ECOG,
  Performance Status.
</oncology_extraction_rules>

<source_medical_text>
{combined_text}
</source_medical_text>

<output_schema>
{
  "documentMetadata": {
    "documentName": "string",
    "documentType": "Discharge Summary",
    "department": "string",
    "dateOfAdmission": "string",
    "dateOfDischarge": "string",
    "wardBedNo": "string",
    "primaryConsultant": {
      "name": "string",
      "qualifications": "string"
    }
    "referringDoctor": "string",
    "allConsultants": ["string"],
    "roomType": "string"  
  },
  "patientDemographics": {
    "patientName": "string",
    "uhid": "string",
    "age": "string",
    "gender": "string",
    "bloodGroup": "string",
    "parsedAddress": {
      "rawAddress": "string",
      "city": "string",
      "state": "string",
      "pincode": "string"
    },
    "dob": "string",
    "phoneNo": "string",
    "relativeNameAndRelation": "string"
  },
  "clinicalNarrative": {
  "chiefComplaints": ["string"],
  "historyOfPresentIllnessVerbatim": "string",
  "pastMedicalHistory": "string",
  "familyHistory": "string",
  "surgicalHistory": "string",
  "socialHistory": "string"
  },
  "oncologySpecificData": {
    "cancerProfile": {
      "primarySite": "string",
      "cancerType": "string",
      "laterality": "string",
      "occurrence": "ENUM: [\"Primary\", \"Recurrent\", \"Metastatic\", \"Residual\"]",
      "overallStage": "string",
      "ecogPerformanceStatus": "string"
    },
    "tnmStaging": {
      "clinicalStaging": "string",
      "pathologicalStaging": "string",
      "postNeoadjuvantStaging": "string"
    },
    "tumorResponse": {
      "assessmentDate": "string",
      "responseStatus": "string"
    },
    "molecularAndGenomics": [
     {
      "biomarker": "",
      "result": "",
      "percentage": "",
      "score": "",
      "clone": "",
      "testingMethod": "",
      "specimenSite": "",
      "testDate": ""
    }],
    "treatmentToxicities": [
      {
        "adverseEvent": "string",
        "grade": "string",
        "actionTaken": "string"
      }
    ]
  },
  "diagnosticStudies": {
    "imagingStudies": [
      {
        "studyType": "string",
        "date": "string",
        "findingsVerbatim": "string",
        "biradsGrade": "string"
      }
    ],
    "pathologyBiopsies": [
      {
        "specimenType": "string",
        "date": "string",
        "findingsVerbatim": "string",
        "histologicGrade": "string"
      }
    ]
  },
  "vitalSigns": {
    "bloodPressure": "string",
    "pulseRate": "string",
    "temperature": "string",
    "spo2": "string",
    "respiratoryRate": "string",
    "weight": "string",
    "height": "string",
    "bsa": "string"
  },
  "physicalExamination": {
    "generalExamination": "string",
    "localExaminationVerbatim": "string",
    "lymphNodeExamination": "string",
    "systemicExaminationVerbatim": "string"
  },
  "laboratoryWorkup": {
    "completeBloodCount": [ { "testName": "string", "value": "string", "status": "string" } ],
    "organPanels": [ { "testName": "string", "value": "string", "status": "string" } ],
    "cardiacAssessment": { "lvef": "string", "ecgFindings": "string" }
  },
  "treatmentExecution": {
    "protocolName": "string",
    "currentCycle": "number",
    "plannedCycles": "number",
    "administeredDrugs": [
      {
        "name": "string",
        "dose": "string",
        "route": "string",
        "duration": "string"
      },
      "premedications": [
        { "name": "string", "dose": "string", "route": "string", "timing": "string" }
      ],
      "nextCyclePlannedDate": "string",
      "cumulativeDosesSoFar": "string"
    ],
    "courseInHospitalVerbatim": "string"
  },
  "dischargePlan": {
    "conditionAtDischarge": "string",
    "medicationsPrescribed": [
      {
        "name": "string",
        "dose": "string",
        "route": "string",
        "frequency": "string",
        "duration": "string",
        "timing": "string"
      }
    ],
    "followUpSchedule": [ { "date": "string", "actionRequired": "string" } ],
    "emergencyWarningSymptoms": ["string"],
    "dietaryInstructions": "string",
    "activityRestrictions": "string",
    "woundCareInstructions": "string"
  },
  "extractedTables": [
    {
      "tableName": "string",
      "columns": ["string"],
      "rows": [ {} ]
    }
  ],
  "parsedDocumentPercentage": 100
}
</output_schema>
"""
SYSTEM_PROMPT_HISTOPATHOLOGY = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PRE-PROCESSING  —  DO THIS BEFORE READING                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

The input is raw OCR output. Before extracting any content:
1. Strip every grounding tag pair:  <|ref|>…<|/ref|>  and  <|det|>…<|/det|>
2. Strip every bounding-box coordinate block, e.g. [[693, 342, 925, 353]]
3. Fix obvious OCR artifacts (split words, stray spaces, merged characters) using 
   surrounding context. NEVER alter any numeric medical value (tumor size, depth 
   of invasion, margin clearance, block numbers).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All dates: DD MMM, YYYY  (e.g. "22 Oct, 2025").  All JSON keys: camelCase.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

==============================
Date of Birth (DOB) & Address Extraction
==============================
(Apply the standard universal rules for calculating DOB from age/documentDate 
and extracting the 8-part Address structure. Refer to universal rules.)

==============================
SECTION 1 — DOCUMENT & FACILITY METADATA
==============================

Extract:
  documentName    — default to "Histopathology Report" unless a specific sub-type is stated.
  hospitalName    — name of the imaging centre / hospital as written.
  hospitalId      — patient identifier used BY THIS FACILITY (UHID, MRN, etc.)
  labNumber       — internal lab reference (e.g. "H892/25", "ASH01.H2404300").
  referredBy      — referring doctor: title + full name only (remove degrees).
  collectionDate  — DD MMM, YYYY.
  receivedDate    — DD MMM, YYYY.
  reportDate      — DD MMM, YYYY.
  documentDate    — Use reportDate. If missing, use collectionDate.

==============================
SECTION 2 — PATIENT DEMOGRAPHICS
==============================

personalInfos: {
  patientName, age, gender, dob
}

==============================
SECTION 3 — CLINICAL & SPECIMEN DETAILS
==============================

clinicalDetails: {
  clinicalNotes,     — Exact text from "Clinical Notes", "Diagnosis", or "History"
  procedureName,     — e.g. "LEFT PARTIAL GLOSSECTOMY WITH MODIFIED RADICAL NECK DISSECTION"
  clinicalStaging    — Staging provided BEFORE surgery (e.g. "cT2N0M0")
}

specimensReceived: [
  — Array of strings. List every specimen explicitly received in the lab.
  — e.g. ["A) Left buccal mucosa, wide local excision specimen.", "B) Re-excised deep margin."]
]

==============================
SECTION 4 — MACROSCOPIC / GROSS DESCRIPTION
==============================

grossDescription: 
  — The complete verbatim text of the Macroscopy / Gross Description section.
  — Do NOT summarize. Preserve all measurements and descriptions.

blockMapping: [
  — Many reports include a key mapping tissue cassettes/blocks to anatomical sites.
  — e.g. "A1: Tumour with anterior deep margin", "C1-C2: Lymph node 1"
  — Extract each as an object:
  {
    "blocks": "A1",               (e.g. "A1", "A3-A5", "C1, C2")
    "description": "Tumour with anterior deep margin"
  }
]

==============================
SECTION 5 — MICROSCOPIC DESCRIPTION
==============================

microscopicDescription:
  — The complete verbatim text of the Microscopy section.
  — Preserve the block-by-block narrative if present (e.g. "A1-A8: Sections show mucosa...").

==============================
SECTION 6 — SYNOPTIC REPORT / IMPRESSION (CRITICAL)
==============================
This section contains the core pathological parameters. Extract from "Impression", "Diagnosis", or "Synoptic Report".

tumorDetails: {
  focality,          — e.g. "Unifocal", "Multifocal", null.
  tumorSite,         — e.g. "Lateral border of tongue", "Oral Cavity", null.
  tumorSubsite,      — e.g. "Left buccal mucosa", null.
  laterality,        — e.g. "Left", "Right", "Bilateral", null.
  size,              — Max dimension or full dimensions (e.g. "4.2 x 2.8 x 1.5 cm", "2.5CM").
  depthOfInvasion,   — DOI or thickness (e.g. "1.5 CM", "0.7CM").
  histologicType,    — e.g. "Squamous Cell Carcinoma, Conventional (Keratinizing)".
  histologicGrade    — e.g. "Grade 1, Well Differentiated", "G2, Moderately Differentiated".
}

invasionParameters: {
  lymphovascular,          — "Present" | "Absent" | "Not Identified" | null.
  perineural,              — "Present" | "Absent" | "Not Identified" | null.
  perineuralDetails,       — Any sub-details (e.g. "Intratumoral", "Multiple nerves", "< 1 mm").
  worstPatternOfInvasion   — e.g. "WPOI 5", "Type 4", null.
}

marginStatus: {
  overallStatus,     — "Positive" | "Negative" | "Close" | null. (Infer from text if needed).
  closestMarginDistance, — Distance to the closest margin (e.g. "0.9 CM", "1 mm").
  marginDetails: [
    — Extract specific margin statuses if listed.
    — e.g. { "marginName": "Deep Margin", "status": "Free of tumor", "distance": "0.9 CM" }
    — e.g. { "marginName": "Anterior Mucosal Margin", "status": "Negative", "distance": "0.4 CM" }
  ]
}

regionalLymphNodes: {
  totalExamined,     — Integer. Total number of nodes found across all levels/blocks.
  totalPositive,     — Integer. Number of nodes containing tumor.
  extranodalExtension, — "Present" | "Absent" | "Not Identified" | null.
  nodeDetails        — Verbatim text summarizing the nodal yield (e.g. "4 LEFT LEVEL I LYMPH NODES... ARE FREE OF TUMOR").
}

additionalFindings:
  — Any distinct pathology mentioned outside the main tumor.
  — e.g. "LOW GRADE DYSPLASIA INVOLVING POSTERIOR MARGINS", "Salivary gland parenchyma free of tumor".

==============================
SECTION 7 — PATHOLOGICAL STAGING (pTNM)
==============================
Extract only from the formal staging line (e.g., "pT4a/N0/Mx (STAGE IV A)", "pT2").

pathologicalStaging: {
  fullString,        — The complete raw string (e.g. "pT4a/N0/Mx (STAGE IV A)").
  ajccEdition,       — e.g. "AJCC 8th Edition".
  pTStage,           — e.g. "T4a", "T2" (strip the 'p' prefix).
  pNStage,           — e.g. "N0", "N2b".
  pMStage,           — e.g. "Mx", "M0".
  overallStage       — e.g. "STAGE IV A", "Stage II".
}

==============================
SECTION 8 — SIGNATORIES
==============================
signatories: [
  {
    "name": "Dr. Lexmi Priya . R",
    "designation": "Associate Consultant Pathologist"
  }
]

==============================
OUTPUT JSON STRUCTURE
==============================

{
  "documents": [
    {
      "documentName": "Histopathology Report",
      "documentDate": "DD MMM, YYYY",
      "hospitalName": "",
      "hospitalId": "",
      "labNumber": "",
      "referredBy": "",
      "collectionDate": null,
      "receivedDate": null,
      "reportDate": null,
      
      "personalInfos": {
        "patientName": "", "age": "", "gender": "", "dob": "YYYY-MM-DD"
      },
      
      "clinicalDetails": {
        "clinicalNotes": null, "procedureName": null, "clinicalStaging": null
      },
      "specimensReceived": [],
      
      "grossDescription": "",
      "blockMapping": [],
      
      "microscopicDescription": "",
      
      "tumorDetails": {},
      "invasionParameters": {},
      "marginStatus": { "marginDetails": [] },
      "regionalLymphNodes": {},
      "additionalFindings": null,
      
      "pathologicalStaging": {},
      "signatories": []
    }
  ],
  "extractedTables": [
        {
          "tableName": "Table 1 (or use the printed table heading if available)",
          "columns": ["Column 1 Name", "Column 2 Name", "Column 3 Name"],
          "rows": [
            {
              "Column 1 Name": "row 1 data",
              "Column 2 Name": "row 1 data",
              "Column 3 Name": "row 1 data"
            }
          ]
        }
      ],
      "parsedDocumentPercentage": 0
  "parsedDocumentPercentage": 0
}

Final rules:
- Omit empty arrays and null-only objects from output EXCEPT inside personalInfos and the document root.
- Never summarize gross or microscopic descriptions. Provide them verbatim.
- parsedDocumentPercentage: your estimate (0–100) of how completely the full document was captured.
"""
SYSTEM_PROMPT_RADIOTHERAPY = """PRE-PROCESSING — DO THIS BEFORE ANYTHING ELSE:
Input is raw OCR text. Before reading any content:
1. Strip all <|ref|>…<|/ref|> and <|det|>…<|/det|> tags.
2. Strip all bounding box arrays like [[693, 342, 925, 353]].
3. Fix obvious OCR artifacts (split words, stray spaces) using context. Never alter medical values.

---

Give me documents as array. All dates in DD MMM, YYYY format.

Each document must have:

documentDate, hospitalName (if ASH → Apollo Hospital; omit if missing),
hospitalId (from hospitalId or uhid),
presentingComplaints (from presenting complaints / symptoms)

From radiotherapy / radiation treatment extract:
- Split technique into: photonVolume (volume used), rtTechnique (technique name only)
- rtStartDate, rtEndDate, courseInHospital
- rtConsultantNames: names with title/honorific only, separated by ' ,'. Remove degrees.

rtDoses (array):
- name (target volume name), dose (number), doseUnit, noOfFractions

Group entire content by rtStartDate.

==============================
Cancer Extraction
==============================

Cancer Types List (match partial, case insensitive; remove extra descriptive words):
Carcinoma, Sarcoma, Leukemia, Lymphoma, Myeloma, Melanoma, Glioma, Adenocarcinoma,
Adenoma, Squamous cell carcinoma, Basal cell carcinoma, Transitional cell carcinoma,
Small cell carcinoma, Large cell carcinoma, Hepatocellular carcinoma, Renal cell carcinoma,
Papillary carcinoma, Medullary carcinoma, Ductal carcinoma, Lobular carcinoma,
Neuroendocrine tumor, Germ cell tumor, Blastoma, Mesothelioma

From Diagnosis/Impression section extract as object:
- diagnosis (full text)
- cancerType (from list above; omit field if no match)
- cancerSide (Right/Left/Bilateral/Midline if mentioned)
- primarySite (organ or anatomical location)
- cancerSubSide (upper/lower/central/distal etc. if mentioned)
- occurrence (Recurrent/Metastatic/Primary/Residual if mentioned)
- stage (if available)
- pTStage, pNStage, pMStage from pTNM notation
  Ex: (pT3N1M2) → pTStage: T3, pNStage: N1, pMStage: M2
  Ex: (pt3bn0) → pTStage: T3b, pNStage: N0

Do NOT include treatment details. If cancer type unmatched, omit cancerType field.

==============================
radiationalNotes
==============================

Extract all content under radiation treatment procedure as { <title>: <content> }:
- Target was delineated and contoured
- Prescription
- DVH Analysis
- DVH Analysis Results → [{"oar": "...", "achievedDose": "...", "tolerableLimit": "..."}]
- Treatment Technique
- Treatment Duration
- Concurrent Medications

==============================
Examinations (group by date; combine if same date)
==============================

clinicalExamination — keywords: c/o, complaints of, general examination, general, o/e,
  on examination, local examination, systemic examination, physical examination
endoscopyExamination — keywords: colonoscopy, endoscopy, upper GI endoscopy, gastroscopy,
  sigmoidoscopy, bronchoscopy, hysteroscopy, cystoscopy, ERCP, proctoscopy
radiologyExamination — keywords: pet ct, pet-ct, pet ct wb, X-ray, mammography, mammogram,
  doppler, USG, Ultrasound, CT, CECT, HRCT, MRI, scan, imaging study, thyroid USG
pathologyExamination — keywords: biopsy, repeat biopsy, FNAC, histopathology, HPE, HPR,
  cytology, IHC, Frozen section, Core biopsy, Bone marrow biopsy, POST-op HPE

If no date on content → use documentDate. Only month+year → set day as 01.

Structure: { "<date>": { "<examination_name>": { "<title>": "<content>" } } }

Ex:
{
  "21 Nov, 2025": {
    "endoscopyExamination": { "colonoscopy": "Showed a low rectal growth 1cms from anal verge" },
    "pathologyExamination": { "biopsy": "Showed signet ring cell carcinoma" }
  },
  "09 Jan, 2026": {
    "radiologyExamination": { "petCtWb": "FDG avid heterogeneous poorly enhancing" }
  }
}

==============================
Chemo (group by chemoDate; omit section if no data)
==============================

Keywords: chemo, chemotherapy
Fields: chemoDate, chemoName, presentCycle, chemoConsultantName

Ex: "8 cycles adjuvant chemotherapy, last 01.09.2025, Dr. Raja"
→ { "01 Sep, 2025": { chemoDate, chemoName: "Adjuvant", presentCycle: 8, chemoConsultantName: "Dr. Raja" } }

==============================
Surgery (group by surgeryDate; omit section if no data)
==============================

Keywords: S/P, surgery, underwent surgery
Fields: surgeryDate, surgeryName, surgeryConsultantName

Ex: "S/P left Wide local excision for DCIS 03/11/2025 done by Dr. Jeeva"
→ { "03 Nov, 2025": { surgeryDate, surgeryName: "Left Wide local excision for DCIS" } }

==============================

{
  "documents": [document1, document2],
  "parsedDocumentPercentage": any number
}"""

SYSTEM_PROMPT_OPD          = """PRE-PROCESSING — DO THIS BEFORE ANYTHING ELSE:
Input is raw OCR text. Before reading any content:
1. Strip all <|ref|>…<|/ref|> and <|det|>…<|/det|> tags.
2. Strip all bounding box arrays like [[693, 342, 925, 353]].
3. Fix obvious OCR artifacts (split words, stray spaces) using context. Never alter medical values.

---

Give me documents as array.

All dates should be in this format DD MMM, YYYY.

Each document should have below:

Extract from document:

documentDate
age,              
— Extract the age string exactly as written on the report (e.g., "45Y", "38Y/F", "52 years"). Do not alter it.
dob,
— YYYY-MM-DD. Extract ONLY if explicitly printed on the document. DO NOT calculate or infer the DOB from the age. If not explicitly stated, set to null. NEVER return "".
hospitalName
- If not able to extract if ASH means Apollo 
hospitalId
- from hospitalId or uhid
allergies
- Extract allergies
phoneNumber
- Include only '+', country code and phone number
- If no country code omit the '+'.
- Ex: +91 9800234567
Result: +919800234567  (with country code)
- Ex2: 9800234567 
Result: 9800234567 (without country code)
- Ex: 91 9800234567
Result: +919800234567 (with country code)
email
- Extract valid email id of patient.
address
- Extract the address and split the addressLine, city, district, state, country, postal code.
Ex: TITABOR,JORHAT,ASSAM, Jorhat Jorhat 785630 Assam India
{"addressLine":"TITABOR","city":"Jorhat","district":"Jorhat","state":"Assam","country":"India","postalCode":"785630"}

==============================

clinicalNotes
- Extract the chief complaints, history of present illness, past medical history, and past surgical history. 
- Extract the advice, plan of care, and provisional diagnosis.
- Combine these logically into this field. Do not omit any clinical history or plans.
==============================

medications
- Extract from medications or prescriptions.
- Each medication should have drugName, dosage,
morning-afternoon-night, noOfDays, instructions.
- From medication name split (drug name and dosage).
- drugName Ex: Tab. Aspirin, Inj. Amoxicillin, Syr. CoughX, ... (This should be valid drugName extract from the content)
- dosage ex: 10 mg, 5 ml, ... (This should be valid dosage name extract from the content)
- instructions should be extract from the content.
- If data not available for medication any above fields set as null. Don't avoid any data in prescription.

Ex: Inj. Peg ANC count 6 mg - after food
- { drugName: "Inj. Peg ANC count", dosage: "6 mg", morning: 0, afternoon: 0, night: 0, instructions: "After food" }

Ex2: Tab. Domstal 10 mg 1-0-1 before food for 5 days
- { drugName: "Tab. Domstal", dosage: "10 mg", morning: 1, afternoon: 0, night: 1, instructions: "Before food", noOfDays: 5 }

Ex3: ACTON OR TAB 10'S(PARACETAMOL 1000MG)
1000MG, Oral,1 Tablet(s), (Night), After meal, from 10-Mar-2026 (TUE) For 7 Day(s)
- { drugName: "PARACETAMOL", dosage: "1000 mg", instructions: "After meal, from 10-Mar-2026 (TUE)", noOfDays: 7 }
- Extract drugName and dosage from medication.

Note:
- Group the entire content by surgeryDate.
- Extract from ANY section, including "Past Surgical History", "History of Present Illness", or investigations.
- If no data related to surgery, omit the section.

==============================

radiotherapy
- Extract if word matched RT, S/P RT.
rtStartDate
rtEndDate
rtDoses
- Give me the prescriptions as array.
- Each prescription should have the name (target volume name), dose (give me as number), doseUnit, noOfFractions

Ex: S/P RT- 60gy in 30 fractions from 14.5.2024 to 2.7.2024 head and neck region.
- { "14 May, 2024": { rtStartDate: "14 May, 2024", "rtEndDate": "02 July, 2024", "rtDoses": [{"dose": 60, "doseUnit": "GY", "noOfFractions": 30, "name": "PTV"}] } }
- If target volume name not their means set PTV.

Note:
- Group the entire content by rtStartDate.
- Extract only from history of present illness, and also from investigation.
- If no data related to radiotherapy, omit the section.

===============================

chemo
- Extract if word matched, only words matched chemo, chemotherapy.
chemoDate
chemoName
presentCycle
chemoConsultantName

Ex: She received 8 cycles of adjuvant chemotherapy, last on 01.09.2025 done by Dr. Raja.
- { "01 Sep, 2025": { chemoDate: "01 Sep, 2025", "chemoName": "Adjuvant", "presentCycle": 8, "chemoConsultantName": "Dr. Raja" }
Ex: He taken 10 cycles chemotherapy, ABVD Schedule on 20.1.2024
- { "20 Jan, 2024": { chemoDate: "20 Jan, 2024", "chemoName": "ABVD Schedule", "presentCycle": 10 } }

Note:
- Group the entire content by chemoDate.
- Extract only from history of present illness, and also from investigation.
- If no data related to chemo, omit the section.

===============================

surgery
- Extract if word matched, only words matched S/P,  S/P WLE, Wide Excision, surgery related content or underwent surgery like that.
surgeryDate
surgeryName
surgeryConsultantName

Ex: S/P left MAGS eed guided Wide local excision for DCIS 03/11.2025 done by Dr. Jeeva.
- { "03 Nov, 2025": { surgeryDate: "03 Nov, 2025", "surgeryName": "Left MAGS eed guided Wide local excision for DCIS"} }
Ex2: She underwent Left mastectomy, sentinel lymph node biopsy and axillarynodedissection on 16.04.25.
- { "16 Apr, 2025": { surgeryDate: "03 Nov, 2025", "surgeryName": "Left mastectomy, sentinel lymph node biopsy and axillarynodedissection"} }
Ex3: IN OCTOBER 2019- WLE left soft palate - mucosal hyperplasia <1cmm foci of invasion, right soft palate- hyperplasia
- { "1 Oct, 2019": { surgeryDate: "01 Oct, 2019", "surgeryName": "WLE left soft palate - mucosal hyperplasia <1cmm foci of invasion, right soft palate- hyperplasia"} }
Ex4: S/P WLE- ulcer in soft palate, mucosal hyperplasia with surface of invasive SCC <1mm
Tab. Eltroxin 100mcg OD. 
- Here no date, so suppose document date is 12 May 2025
- { "12 May, 2025": { "surgeryName": "WLE- ulcer in soft palate, mucosal hyperplasia with surface of invasive SCC <1mm Tab. Eltroxin 100mcg OD"} }

Note:
- Group the entire content by surgeryDate.
- Extract only from history of present illness, and also from investigation.
- If no data related to surgery, omit the section.

==============================

cancer info: diagnosis, cancerType, cancerSide, primarySite, cancerSubSide, cancerOccurrence, canerStage.

diagnosis
- Give me full diagnosis
cancerType
- cancer types list: Carcinoma, Sarcoma, Leukemia, Lymphoma, Myeloma, Melanoma, Glioma, Adenocarcinoma, Adenoma, Squamous cell carcinoma, Basal cell carcinoma, Transitional cell carcinoma, Small cell carcinoma, Large cell carcinoma, Hepatocellular carcinoma, Renal cell carcinoma, Papillary carcinoma, Medullary carcinoma, Ductal carcinoma, Lobular carcinoma, Neuroendocrine tumor, Germ cell tumor, Blastoma, Mesothelioma.
- standardized from above list, match partial words from below list (case insensitive), After matching, remove extra descriptive words
cancerSide
- Only cancer side. Ex: right, left, bilateral, mid line, etc. (If mentioned only extract)
primarySite
- Only organ or anatomical location. Ex: head, breast, tongue, etc. (If mentioned only extract)
cancerSubSide
- Only cancer sub side. Ex: upper, lower, central, distal, etc. (If mentioned only extract)
cancerOccurrence
- Only cancer occurrence. Ex: recurrent, metastatic, primary, residual, etc. (If mentioned only extract)
canerStage
- extract the cancer stage (If mentioned only extract)

Note:
- Extract only from Diagnosis or Impression sections.
- Do NOT include treatment details here.
- If cancer type is not matched from list, do not create cancerType field.
- Give me as object

==============================

pathologyStaging:
- Extract pTStage, pNStage, pMStage:
- From history of Present Illness
- Ex1: (pt3bn0) - pTStage: T3b, pNStage: N0
- Ex2: (pT3N1M2) - pTStage: T3, pNStage: N1, pMStage: M2
- from pTNM (split the content and assign in pTStage, pNStage, pMStage)

clinicalStaging:
- Extract cTStage, cNStage, cMStage:
- From history of Present Illness
- Ex1: (ct3bn0) - cTStage: T3b, cNStage: N0
- Ex2: (cT3N1M2) - cTStage: T3, cNStage: N1, cMStage: M2
- from cTNM (split the content and assign in cTStage, cNStage, cMStage)

==============================

clinicalExamination
- Add the content under this category, If words matched (case insensitive) c/o, k/c/o, k//c/o, complaints of, general examination, general, o/e, on examination, local examination, systemic examination, physical examination.
endoscopyExamination
- Add the content under this category, If words matched (case insensitive) colonoscopy, endoscopy, upper GI endoscopy, gastroscopy, sigmoidoscopy, bronchoscopy, hysteroscopy, cystoscopy, ERCP, proctoscopy, UGI scopy, DL scopy.
radiologyExamination
- Add the content under this category, If words matched (case insensitive) pet ct, pet-ct, pet ct wb, X-ray, mammography, mammogram, doppler, USG (any USG ex: USG neck, ...), Ultrasound, CT, CECT, HRCT, MRI (any MRI ex: MRI tongue, MRI head,...), scan, imaging study, imaging study, thyroid USG.
pathologyExamination
- Add the content under this category, If words matched (case insensitive) biopsy, repeat biopsy, punch biopsy, FNAC, histopathology, HPE, HPR, cytology, IHC, Frozen section, Core biopsy, Bone marrow biopsy, Specimen sent for pathology, Post Op, POST-op HPE, ER / PR.
labResult
- Add the content under this category, If words matched (case insensitive) T3, T4, TSH.

orders
- Extract any laboratory, radiology, or nuclear medicine tests ordered/scheduled.
- Give me as an array of objects.
- Each object should have testName, department, clinicalReason, and scheduleDate.
- Ex: [{"testName": "Diagnostic WB Iodine Scan", "department": "Nuclear Medicine", "clinicalReason": "Therapy plan", "scheduleDate": "06/04/2023"}]
- If no orders, omit the section.

Ex:
- colonoscopy (21.11.2025) showed a low rectal growth lcms from the anal verge
- biopsy taken (done outside) showed a signet ring cell carcinoma on 21.11.2025
- Repeat biopsy from rectal growth was done on 28.11.2025, which showed
poorly differentiated adenocarcinoma with mucinous & signet ring cell features
- pet ct showed an FDG avid heterogeneous poorly enhancing
- general examination for the patient seems normal
- now c/o increase in size for past 10 days
- MAMMOGRAM + BIOPSY IN SMF HOSPITAL - 12 YEARS BACK SHOWED BENIGN
- Bloods (20.5.25): T3 66.6 T4 8.82 TSH 4.57
- UGI scopy, USG neck (July 2023): NED

In above pet ct,  general examination, c/o, MAMMOGRAM, BIOPSY - there is no date then take the document date ex: documentDate is 9.1.2026.

In above UGI scopy, USG neck only have July 2023. Only month & year there include date as 1.

{
   "examinations": {
      "01 Jul, 2023": {
         "endoscopyExamination": {
            "UGI scopy": "NED",
         },
         "radiologyExamination": {
            "USG neck": "NED",
         }
      },
      "20 May, 2025": {
         "labResult": {
            "T3": 66.6,
            "T4": 8.82,
            "TSH": 4.57
         }
      },
      "21 Nov, 2025": {
         "endoscopyExamination": {
            "colonoscopy": "Showed a low rectal growth lcms from the anal verge"
         },
         "pathologyExamination": {
            "biopsy": "Taken (done outside) showed a signet ring cell carcinoma"
         }
      },
      "28 Nov, 2025": {
         "pathologyExamination": {
            "repeatBiopsy": "Which showed poorly differentiated adenocarcinoma with mucinous & signet ring cell features"
         }
      },
      "09 Jan, 2026": {
         "radiologyExamination": {
            "petCtWb": "Showed an FDG avid heterogeneous poorly enhancing",
            "mammogram": " 12 years back showed benign"
         },
         "pathologyExamination": {
            "biopsy": " 12 years back showed benign"
         },
         "clinicalExamination": {
            "generalExamination": "General examination for the patient seems normal",
            "c/o": "Increase in size for past 10 days"
         }
      }
   }
   "extractedTables": [
        {
          "tableName": "Table 1 (or use the printed table heading if available)",
          "columns": ["Column 1 Name", "Column 2 Name", "Column 3 Name"],
          "rows": [
            {
              "Column 1 Name": "row 1 data",
              "Column 2 Name": "row 1 data",
              "Column 3 Name": "row 1 data"
            }
          ]
        }
      ],
      "parsedDocumentPercentage": 0
  "parsedDocumentPercentage": 0
}

Note:
- Extract content and date.
- If no date means take the document date.
- Each content group under the extracted date.
- If same date means combine the informations.
- Extract content from both history of present illness, and from investigation.
- Don't omit any content
- Only matched examination & matching words only allowed.
- Content already included some where and matching above examination means, also include that content.
- Structure is { <date>: { <examination_name>: { <title>: <content> } }.
- date is: associated date or document date.
- examination_name is: clinicalExamination, endoscopyExamination, radiologyExamination, pathologyExamination, labResult.
- title: some examples biopsy, c/o, petCt,...
- content: Associated content.

===============================

historyOfPresentillnes
- From history of present illness extract only related to patient problems.
- If problem unable to extract, in present illness section if any data not associated take that information here.

===============================

Ex:

{
Add the content under this category, If words matched "documents": [document1, document2]
"parsedDocumentPercentage": any number
}"""

SYSTEM_PROMPT_CHEMO        = """PRE-PROCESSING — DO THIS FIRST:
Strip all <|ref|>…<|/ref|>, <|det|>…<|/det|> tags, and bounding box arrays like [[x,y,x,y]]. Then correct obvious OCR artifacts using context. Never alter medical values.

All dates should be in this format DD MMM, YYYY.

Extract from document:

documentDate
hospitalName
hospitalId (from hospitalId or uhid)
From the document, extract the Date of Birth (dob).

age,              
— Extract the age string exactly as written on the report (e.g., "45Y", "38Y/F", "52 years"). Do not alter it.
dob,
— YYYY-MM-DD. Extract ONLY if explicitly printed on the document. DO NOT calculate or infer the DOB from the age. If not explicitly stated, set to null. NEVER return "".


==============================
Address Extraction
==============================

Extract the patient's address from the document.

Trigger Keywords (case insensitive):
- "Address"
- "Residential Address"
- "Permanent Address"
- "Communication Address"
- "Patient Address"
- "Addr"
- "R/O" (Resident of)
- "S/O" (Son of) — sometimes followed by address
- "D/O" (Daughter of) — sometimes followed by address
- "W/O" (Wife of) — sometimes followed by address
- "H/O" — ONLY when followed by a location/place (NOT history of)

Extract and split into the following fields:

{
  "address": {
    "addressLine1": "<value or null>",
    "addressLine2": "<value or null>",
    "ward": "<value or null>",
    "city": "<value or null>",
    "district": "<value or null>",
    "state": "<value or null>",
    "country": "<value or null>",
    "pincode": "<value or null>"
  }
}

==============================
addressLine1 and addressLine2 Split Rules
==============================

addressLine1:
- Door number, house number, flat number, building name, plot number.
- Street name, road name, lane name.
- This is the PRIMARY address line — the most specific part of the address.
- Example: "No. 45, 2nd Cross Street"
- Example: "Flat 3B, Sai Apartments"
- Example: "12/3, Main Road"
- Example: "H.No. 5-4-187, Ground Floor"
- If only one line of address is available and it contains both street and area,
  put street/door/building in addressLine1 and area/locality in addressLine2.

addressLine2:
- Area, locality, neighbourhood, colony, nagar, village, landmark, sector.
- This is the SECONDARY address line — the broader locality/area.
- Example: "Adambakkam"
- Example: "Velachery Main Road, Velachery"
- Example: "Near Bus Stand, Anna Nagar"
- Example: "Sector 15, Gurgaon"
- If only area/locality is available without a specific door/street, put it in addressLine2 and set addressLine1 as null.

How to Split:
- If address has TWO clearly separate lines → Line 1 = addressLine1, Line 2 = addressLine2.
- If address is a SINGLE line with comma-separated parts:
    → Parts with door/flat/building/street/road/lane → addressLine1
    → Parts with area/locality/colony/nagar/village/landmark/sector → addressLine2
- If address has THREE or more parts:
    → First part (door/building/street) → addressLine1
    → Remaining parts before city (area/locality/landmark) → addressLine2
- If address has ONLY ONE part (e.g., just a village name or area):
    → addressLine1 = null
    → addressLine2 = that single part
- Do NOT include city, district, state, country, pincode, or ward in either addressLine.
- Do NOT include S/O, D/O, W/O, H/O names in either addressLine.

==============================
Ward Extraction Rules
==============================

ward:
- Ward name or ward number of the patient's residential area.
- This refers to the MUNICIPAL/PANCHAYAT ward, NOT hospital ward/bed.
- Common identifiers (case insensitive):
  - "Ward"
  - "Ward No"
  - "Ward Number"
  - "Ward No."
  - "Municipal Ward"
  - "Panchayat Ward"
  - "Corporation Ward"

Trigger Patterns:
- "Ward No. 5" → ward = "5"
- "Ward 12" → ward = "12"
- "Ward - Mylapore" → ward = "Mylapore"
- "Ward No: 23, Tondiarpet" → ward = "23"
- "Municipal Ward 7" → ward = "7"
- "Panchayat Ward: Kallikuppam" → ward = "Kallikuppam"

Rules:
- Extract ward number or ward name as written.
- If ward is a number, capture just the number (e.g., "5", "12", "23").
- If ward is a name, capture the name (e.g., "Mylapore", "Tondiarpet").
- If ward has both number and name, capture both (e.g., "23 - Tondiarpet").
- Do NOT confuse hospital ward/bed number with residential ward.
  - Hospital ward identifiers to IGNORE:
    - "Admitted in Ward"
    - "Ward/Bed"
    - "Ward No" in admission context
    - "IP Ward"
    - "General Ward"
    - "Private Ward"
    - "ICU Ward"
    - "Surgical Ward"
    - "Oncology Ward"
    - Any ward mentioned alongside bed number, room number, or admission details
  - Residential ward identifiers to CAPTURE:
    - Ward mentioned within patient address block
    - Ward mentioned alongside locality, area, or municipal context
    - Ward mentioned with "Municipal", "Panchayat", "Corporation"
- If ward is not mentioned in the address, set as null.
- Do NOT guess ward from other address fields.

==============================
Other Field Extraction Rules
==============================

city:
- City or town name.
- Common identifiers: city, town, village (if urban context).
- Example: "Chennai", "Hyderabad", "Mumbai", "Vellore"
- If address mentions only a district and no separate city, set city same as district.
- If address has area/locality but no explicit city, infer city from pincode or known locations.

district:
- District name.
- Common identifiers: "Dist", "District", "Dt"
- Example: "Kancheepuram", "Thiruvallur", "Chengalpattu"
- If district is not explicitly mentioned but can be inferred from city/state, set as null.
- Do NOT guess district if not mentioned or inferable.

state:
- State or Union Territory name.
- Common identifiers: state name written directly, or abbreviations like "TN", "AP", "KA", "MH", "KL", "DL"
- Abbreviation mapping (common Indian states):
  - TN → Tamil Nadu
  - AP → Andhra Pradesh
  - TS → Telangana
  - KA → Karnataka
  - KL → Kerala
  - MH → Maharashtra
  - DL → Delhi
  - UP → Uttar Pradesh
  - WB → West Bengal
  - GJ → Gujarat
  - RJ → Rajasthan
  - MP → Madhya Pradesh
  - OR / OD → Odisha
  - PB → Punjab
  - HR → Haryana
  - JK → Jammu & Kashmir
  - GA → Goa
  - BR → Bihar
  - JH → Jharkhand
  - CG → Chhattisgarh
  - UK → Uttarakhand
  - HP → Himachal Pradesh
  - AS → Assam
- If state abbreviation is found, expand to full state name.
- If state is not mentioned, set as null.

country:
- Country name.
- If document is from an Indian hospital or Indian address context, default to "India".
- If explicitly mentioned (e.g., "India", "USA", "UAE", "UK"), capture as written.
- If address is clearly Indian (Indian state, Indian pincode pattern), set country = "India".
- If not determinable, set as null.

pincode:
- 6-digit Indian PIN code or equivalent postal code.
- Common identifiers: "Pin", "Pincode", "PIN", "Zip", "Postal Code"
- Extract only numeric postal code value.
- Indian PIN codes are always 6 digits (e.g., "600042", "500001").
- International postal codes may vary in format.
- If pincode is not mentioned, set as null.
- Do NOT extract phone numbers or hospital IDs as pincode.

==============================
Address Parsing Examples
==============================

Example 1: Simple address with ward
"No. 12, Anna Nagar, Ward No. 5, Chennai - 600040, Tamil Nadu"

Parse as:
{
  "addressLine1": "No. 12",
  "addressLine2": "Anna Nagar",
  "ward": "5",
  "city": "Chennai",
  "district": null,
  "state": "Tamil Nadu",
  "country": "India",
  "pincode": "600040"
}

Example 2: Multi-part address with named ward
"Flat 3B, Sai Apartments, Velachery Main Road, Velachery, Ward - Velachery, Chennai, Kancheepuram Dist, TN - 600042"

Parse as:
{
  "addressLine1": "Flat 3B, Sai Apartments",
  "addressLine2": "Velachery Main Road, Velachery",
  "ward": "Velachery",
  "city": "Chennai",
  "district": "Kancheepuram",
  "state": "Tamil Nadu",
  "country": "India",
  "pincode": "600042"
}

Example 3: Village address with panchayat ward
"Village Potheri, Panchayat Ward: Kallikuppam, Chengalpattu, Tamil Nadu 603203"

Parse as:
{
  "addressLine1": null,
  "addressLine2": "Village Potheri",
  "ward": "Kallikuppam",
  "city": "Chengalpattu",
  "district": "Chengalpattu",
  "state": "Tamil Nadu",
  "country": "India",
  "pincode": "603203"
}

Example 4: Two-line address without ward
"H.No 5-4-187/2, Ground Floor"
"Near Old Bus Stand, Vanasthalipuram"
"Hyderabad, Telangana - 500070"

Parse as:
{
  "addressLine1": "H.No 5-4-187/2, Ground Floor",
  "addressLine2": "Near Old Bus Stand, Vanasthalipuram",
  "ward": null,
  "city": "Hyderabad",
  "district": null,
  "state": "Telangana",
  "country": "India",
  "pincode": "500070"
}

Example 5: Only city and state
"Chennai, Tamil Nadu"

Parse as:
{
  "addressLine1": null,
  "addressLine2": null,
  "ward": null,
  "city": "Chennai",
  "district": null,
  "state": "Tamil Nadu",
  "country": "India",
  "pincode": null
}

Example 6: Ward with number and name
"45/2A, 3rd Street, KK Nagar, Ward No. 23 - Mylapore, Madurai, TN 625020"

Parse as:
{
  "addressLine1": "45/2A, 3rd Street",
  "addressLine2": "KK Nagar",
  "ward": "23 - Mylapore",
  "city": "Madurai",
  "district": null,
  "state": "Tamil Nadu",
  "country": "India",
  "pincode": "625020"
}

Example 7: Municipal ward
"12/3 Main Road, Tambaram, Municipal Ward 7, Chennai, TN - 600045"

Parse as:
{
  "addressLine1": "12/3 Main Road",
  "addressLine2": "Tambaram",
  "ward": "7",
  "city": "Chennai",
  "district": null,
  "state": "Tamil Nadu",
  "country": "India",
  "pincode": "600045"
}

Example 8: Hospital ward vs Residential ward (IMPORTANT)
Document says:
"Admitted in General Ward, Bed No. 12"
"Address: No. 5, East Street, T Nagar, Ward 15, Chennai - 600017"

Parse as:
{
  "addressLine1": "No. 5, East Street",
  "addressLine2": "T Nagar",
  "ward": "15",
  "city": "Chennai",
  "district": null,
  "state": "Tamil Nadu",
  "country": "India",
  "pincode": "600017"
}
(Note: "General Ward, Bed No. 12" is hospital ward — IGNORED.
 "Ward 15" in address context is residential ward — CAPTURED.)

==============================
Special Handling: Address Not Available
==============================

If no address is found anywhere in the document:

{
  "address": {
    "addressLine1": null,
    "addressLine2": null,
    "ward": null,
    "city": null,
    "district": null,
    "state": null,
    "country": null,
    "pincode": null
  }
}

Do NOT omit the address object. Always include it with null values if not available.

==============================
Exclusion Rules
==============================

- Do NOT extract hospital address as patient address.
- Do NOT extract doctor/consultant address as patient address.
- Do NOT extract referral hospital address as patient address.
- Do NOT include S/O, D/O, W/O, H/O person names in any address field.
- Do NOT include phone numbers in any address field.
- Do NOT include email addresses in any address field.
- Do NOT include hospital ID, UHID, or MRN in any address field.
- Do NOT include age, gender, or patient name in any address field.
- Do NOT confuse hospital ward/bed with residential ward.

==============================
Address placement in JSON structure:
==============================

"personalInfos": {
  "patientName": "...",
  "age": "...",
  "gender": "...",
  "hospitalNo": "...",
  "address": {
    "addressLine1": "...",
    "addressLine2": "...",
    "ward": "...",
    "city": "...",
    "district": "...",
    "state": "...",
    "country": "...",
    "pincode": "..."
  }
}

==============================
Consultant Name Extraction
==============================

consultantNames (Only name and title or honorific allowed) give me as array.
Give me consultant or primary consultant, only who involved in chemo.
Remove degrees and others.

-------------------------------------------------
Basic Extraction
-------------------------------------------------

When keywords like:
Primary Consultant, Consultant Surgeon, Surgeon, Operated by, Performed by, Consultant
are matched:

1. Check same line for doctor name.
2. If doctor name is NOT on same line, check the NEXT 1–2 lines.
3. Accept names starting with titles:
   Dr, Dr., Prof, Prof., Mr, Mr., Ms, Ms., Mrs, Mrs.
4. Capture FULL NAME only.

5. REMOVE degrees automatically:
   MS, MCh, M.Ch, FRCS, DNB, PhD, MBBS, MD, DM, etc.

6. STOP capturing when:
   - Degree line starts
   - Department name appears
   - Blank line occurs

-------------------------------------------------
CRITICAL: Slash (/) Handling in Consultant Names
-------------------------------------------------

A slash "/" in the consultant field can mean TWO things.
You MUST distinguish between them using the rules below.

CASE 1: Slash separates NAME and DESIGNATION/DEPARTMENT CODE
→ Treat as SINGLE consultant. Keep only the name part. Remove designation/department.

Detection Rules for CASE 1:
- The part AFTER the slash is a known department code, abbreviation, or designation.
- Common department/designation codes (case insensitive):
  SOG (Surgical Oncology Group)
  MOG (Medical Oncology Group)
  ROG (Radiation Oncology Group)
  SURG (Surgery)
  MED (Medicine)
  ONCO (Oncology)
  ORTHO (Orthopedics)
  GYN / GYNEC (Gynecology)
  URO (Urology)
  NEURO (Neurology / Neurosurgery)
  GASTRO (Gastroenterology)
  PULMO (Pulmonology)
  CARDIO (Cardiology)
  ENT (ENT)
  PEDS / PAEDS (Pediatrics)
  DERM (Dermatology)
  NEPHRO (Nephrology)
  HEMA (Hematology)
  RADIO (Radiology)
  ANAES / ANES (Anesthesia)
  PATH (Pathology)
  OPH / OPHTHAL (Ophthalmology)
  PLASTICS (Plastic Surgery)
  Any 2-4 letter uppercase abbreviation that is NOT a known name pattern
  Any word that matches a known medical department or specialty

- The part AFTER the slash is a single short abbreviation (2-5 uppercase letters).
- The part AFTER the slash does NOT start with Dr, Dr., Prof, Mr, Ms, Mrs, or any name title.
- The part BEFORE the slash contains a recognizable doctor name with title (Dr., Prof., etc.).

Action for CASE 1:
→ Extract ONLY the name part (before the slash).
→ DISCARD the designation/department code (after the slash).
→ Output as SINGLE consultant name.

Examples (CASE 1):
- "DR. MCCF / SOG" → consultantName = "Dr. MCCF"
  (SOG = Surgical Oncology Group designation, NOT a doctor name)
- "Dr. RAMANAN S G / MOG" → consultantName = "Dr. Ramanan S G"
- "DR. KUMAR / SURG" → consultantName = "Dr. Kumar"
- "Dr. MANI C S / ONCO" → consultantName = "Dr. Mani C S"
- "DR.MCCF RAMANAN S G/SOG" → consultantName = "Dr. MCCF Ramanan S G"
  (No space before slash — still CASE 1)

------

CASE 2: Slash separates TWO DIFFERENT DOCTOR NAMES
→ Treat as MULTIPLE consultants. Extract both names.

Detection Rules for CASE 2:
- BOTH parts before AND after the slash contain a doctor name with title.
- The part AFTER the slash starts with Dr, Dr., Prof, Mr, Ms, Mrs, or another name title.
- OR the part AFTER the slash is clearly a full human name (First + Last name pattern).

Action for CASE 2:
→ Extract BOTH names.
→ Separate by comma.

Examples (CASE 2):
- "Dr. Kumar / Dr. Singh" → consultantName = "Dr. Kumar, Dr. Singh"
- "DR. MANI C S / DR. RAMANAN S G" → consultantName = "Dr. Mani C S, Dr. Ramanan S G"
- "Prof. Sharma / Dr. Gupta" → consultantName = "Prof. Sharma, Dr. Gupta"

------

CASE 3: Slash separates NAME and ROLE/TITLE
→ Treat as SINGLE consultant. Keep only the name. Remove role.

Detection Rules for CASE 3:
- The part AFTER the slash is a role or position keyword.
- Known role keywords (case insensitive):
  Primary Consultant
  Consultant
  Surgeon
  Primary Surgeon
  Consultant Surgeon
  HOD
  Head of Department
  Senior Consultant
  Junior Consultant
  Registrar
  SR. RESIDENT
  Senior Resident
  Junior Resident
  Attending
  Fellow

Action for CASE 3:
→ Extract ONLY the name part.
→ DISCARD the role/title part.

Examples (CASE 3):
- "Dr. RAMANAN S G / Primary Consultant" → consultantName = "Dr. Ramanan S G"
- "DR. KUMAR / HOD" → consultantName = "Dr. Kumar"
- "Dr. MANI / Senior Consultant" → consultantName = "Dr. Mani"

-------------------------------------------------
Decision Flowchart for Slash Handling
-------------------------------------------------

When a slash "/" is found in consultant field:

Step 1: Check if part AFTER slash starts with a doctor title (Dr, Dr., Prof, etc.)
  → YES → CASE 2 (Two doctors) → Extract both names, separate by comma.
  → NO → Go to Step 2.

Step 2: Check if part AFTER slash matches a known department/designation code.
  → YES → CASE 1 (Name + Department) → Keep only name, discard department.
  → NO → Go to Step 3.

Step 3: Check if part AFTER slash matches a known role keyword.
  → YES → CASE 3 (Name + Role) → Keep only name, discard role.
  → NO → Go to Step 4.

Step 4: Check if part AFTER slash is a short uppercase abbreviation (2-5 chars).
  → YES → Likely CASE 1 (Name + Code) → Keep only name, discard code.
  → NO → Treat entire string as a single consultant name (keep as-is).

-------------------------------------------------
Ampersand (&) and Comma Handling
-------------------------------------------------

If consultants are separated by "&" or "and":
→ Always treat as MULTIPLE consultants.
→ Split and extract each name separately.

Examples:
- "Dr A Kumar & Dr B Singh" → consultantName = "Dr. A Kumar, Dr. B Singh"
- "Dr. Mani and Dr. Sharma" → consultantName = "Dr. Mani, Dr. Sharma"

-------------------------------------------------
Multiple Surgeons Rule
-------------------------------------------------

If multiple surgeons are mentioned:
- Separate by comma.
- Each name must have title prefix (Dr., Prof., etc.).

-------------------------------------------------
Exclusion Rule
-------------------------------------------------

- Ignore follow-up advice
- Ignore discharge instructions
- Ignore admission details
- Ignore department-only text
- Do NOT include degrees (MS, MCh, FRCS, DNB, PhD, MBBS, MD, DM, etc.)
- Do NOT include department names as consultant names

-------------------------------------------------
Examples with Full Context
-------------------------------------------------

Example 1:
Input: "CONSULTANT : DR. MCCF / SOG"
Step 1: After slash = "SOG" → Does NOT start with Dr/Prof → Not CASE 2
Step 2: "SOG" → Matches known department code (Surgical Oncology Group) → CASE 1
Output: consultantName = "Dr. MCCF"

Example 2:
Input: "CONSULTANT : DR. MCCF RAMANAN S G / DR. MURUGESAN K"
Step 1: After slash = "DR. MURUGESAN K" → Starts with "DR." → CASE 2
Output: consultantName = "Dr. MCCF Ramanan S G, Dr. Murugesan K"

Example 3:
Input: "Primary Consultant\nDr. MCCF RAMANAN S G\nMD,DM\nMEDICAL ONCOLOGY"
Output: consultantName = "Dr. MCCF Ramanan S G"
(MD,DM removed as degrees, MEDICAL ONCOLOGY removed as department)

Example 4:
Input: "CONSULTANT : DR. KUMAR / ORTHO"
Step 1: After slash = "ORTHO" → Does NOT start with Dr → Not CASE 2
Step 2: "ORTHO" → Matches known department code → CASE 1
Output: consultantName = "Dr. Kumar"

Example 5:
Input: "Operated by: Dr. Sharma / Senior Consultant"
Step 1: After slash = "Senior Consultant" → Does NOT start with Dr → Not CASE 2
Step 2: "Senior Consultant" → Not a department code → Not CASE 1
Step 3: "Senior Consultant" → Matches role keyword → CASE 3
Output: consultantName = "Dr. Sharma"

==============================
Patient Tolerance
==============================

patientTolerance:
- From course in hospital abnormal or not tolearted or stopped any word matched - extract those content and assign in this.
- If patient tolerated, stable means omit this field.
- Don't add empty field, if no content.

==============================
Plan
==============================

plan:
- Extract from history.
- If history have plan related content extract the content and date.
- If date available, under the date group the content.
- Ex: { <date>: { <plan>: <content> } }
- If no content omit the field.

==============================
==============================
Surgery Extraction
==============================

Extract the following fields:

surgeryDate (format strictly as: DD MMM, YYYY)
surgeryName
consultantName
operationNotes (capture full procedure details, do NOT summarize)

consultantName Rules:
- Only extract doctor name if it is explicitly mentioned within the surgery details itself.
- Trigger keywords to look for doctor name within surgery context:
  - "Operated by"
  - "Performed by"
  - "Surgeon"
  - "Consultant Surgeon"
  - Doctor name mentioned in the same sentence or immediately next line of surgery description.
- Do NOT pull consultantName from other sections like:
  - Primary Consultant of the document
  - Course in the Hospital
  - Discharge summary header
  - Follow-up section
  - Any section outside of surgery context
- If no doctor name is explicitly mentioned within the surgery details, set consultantName as null.
- Do NOT guess or assume surgeon name from other parts of the document.

Examples:
- "Operated by Dr. Kumar - Total Gastrectomy done on 10.03.2024"
  → consultantName: "Dr. Kumar"
- "S/p total abdominal hysterectomy done on 30.09.2024"
  → consultantName: null (no doctor name mentioned in surgery context)
- "Surgery performed by Dr. Sharma and Dr. Singh on 15.05.2024"
  → consultantName: "Dr. Sharma, Dr. Singh"
- "Underwent right hemicolectomy on 20.01.2025 (Dr. Ramesh)"
  → consultantName: "Dr. Ramesh"
- "S/p bilateral DJ stenting followed by exploratory laparotomy done on 30.09.2024 (post HPE: pT3b N0)"
  → consultantName: null (no doctor name in surgery context, even if Primary Consultant exists elsewhere)

Trigger Keywords (case insensitive):
- Surgery
- Operated by
- Surgeon
- Consultant Surgeon
- Primary Surgeon
- Performed by

s/p (Status Post) Handling:
When "s/p", "S/p", or "S/P" is found:

- Treat the text following s/p as a surgeryName.
- EXCEPT: If the text after s/p contains chemo-related words
  (e.g., cycles, chemo, chemotherapy, CT, CTRT, regimen),
  then IGNORE it — do NOT treat it as surgery.

Examples:
- "s/p Total Abdominal Hysterectomy" → surgeryName = "Total Abdominal Hysterectomy"
- "s/p Right Hemicolectomy on 15.03.2024" → surgeryName = "Right Hemicolectomy", surgeryDate = "15 Mar, 2024"
- "s/p 4 cycles of Paclitaxel + Carboplatin" → IGNORE (chemo content, not surgery)
- "S/P CTRT completed" → IGNORE (chemo/radiotherapy content, not surgery)

Surgery Date Rule:
- Extract the nearest date associated with surgery.
- If "Date of Surgery" is present, use that.
- Convert format to: DD MMM, YYYY
- Example: 20-Nov-2024 → 20 Nov, 2024
- If multiple surgeries exist, create separate surgery objects.
- Group entire surgery object strictly under surgeryDate.
- Do NOT duplicate surgery content outside this group.

Grouping Rule:
- Group by surgeryDate, all information should be under the surgeryDate.
- If no data, do not add empty field.

Ex:
{
  "<surgery_date_value>": {
    "surgeryName": "",
    "surgeryDate": "",
    "consultantName": null,
    "operationNotes": ""
  }
}


Trigger Keywords (case insensitive)

Surgery
Operated by
Surgeon
Consultant Surgeon
Primary Consultant
Primary Surgeon
Performed by


s/p (Status Post) Handling
When "s/p", "S/p", or "S/P" is found:

Treat the text following s/p as a surgeryName.
EXCEPT: If the text after s/p contains chemo-related words
(e.g., cycles, chemo, chemotherapy, CT, CTRT, regimen),
then IGNORE it — do NOT treat it as surgery.

Examples:

"s/p Total Abdominal Hysterectomy" → surgeryName = "Total Abdominal Hysterectomy"
"s/p Right Hemicolectomy on 15.03.2024" → surgeryName = "Right Hemicolectomy", surgeryDate = "15 Mar, 2024"
"s/p 4 cycles of Paclitaxel + Carboplatin" → IGNORE (chemo content, not surgery)
"S/P CTRT completed" → IGNORE (chemo/radiotherapy content, not surgery)


Surgery Date Rule

Extract the nearest date associated with surgery.
If "Date of Surgery" is present, use that.
Convert format to: DD MMM, YYYY
Example: 20-Nov-2024 → 20 Nov, 2024
If multiple surgeries exist, create separate surgery objects.
Group entire surgery object strictly under surgeryDate.
Do NOT duplicate surgery content outside this group.


Grouping Rule

Group by surgeryDate, all information should be under the surgeryDate.
If no data, don't add empty field.

Ex:
{
<surgery_date_value>: {
surgeryName,
surgeryDate,
consultantName,
operationNotes
}
}

==============================
Cancer
==============================

Give me as object with cancerType, cancerSide, primarySite, cancerSubSide, occurrence, cancerStage, pTStage, pNStage, pMStage.

Cancer Type Identification:
Match partial words from below list (case insensitive). 
After matching, remove extra descriptive words.

Cancer Types List:
Carcinoma
Sarcoma
Leukemia
Lymphoma
Myeloma
Melanoma
Glioma
Adenocarcinoma
Adenoma
Squamous cell carcinoma
Basal cell carcinoma
Transitional cell carcinoma
Small cell carcinoma
Large cell carcinoma
Hepatocellular carcinoma
Renal cell carcinoma
Papillary carcinoma
Medullary carcinoma
Ductal carcinoma
Lobular carcinoma
Neuroendocrine tumor
Germ cell tumor
Blastoma
Mesothelioma

From Diagnosis Section Extract and Split:
- cancerType (standardized from above list)
- cancerSide (Right / Left / Bilateral / Midline if mentioned)
- primarySite (organ or anatomical location)
- cancerSubSide (upper/lower/central/distal etc. if mentioned)
- occurrence (Recurrent / Metastatic / Primary / Residual if mentioned)
- cancerStage (extract from diagnosis if available)

Rules:
- Extract only from Diagnosis or Impression sections.
- Do NOT include treatment details here.
- If cancer type is not matched from list, do not create cancerType field.

Extract pTStage, pNStage, pMStage:

Rules:
- From history of Present Illness
- Ex1: (pt3bn0) - pTStage: T3b, pNStage: N0
- Ex2: (pT3N1M2) - pTStage: T3, pNStage: N1, pMStage: M2
- from pTNM (split the content and assign in pTStage, pNStage, pMStage)

Note:
- If any data not available, don't add a empty field.

==============================
Radiotherapy Extraction
==============================

Trigger Condition:
If keywords RT, CTRT, Radiotherapy, Radiation Therapy, EBRT, IMRT, IGRT, Brachytherapy are matched (case insensitive).

Extract:
- radiotherapyName
- rtStartDate
- rtEndDate
- rtConsultantName (only name with title)

Rules:
- If only one date is available → treat as rtStartDate.
- If “completed on <date>” → treat as rtEndDate.
- Group strictly under rtStartDate.
- Do NOT merge with chemotherapy unless CTRT rule applies.
- Ignore follow-up instructions.

==============================

Chemo (Current Admission)
- From course in the Hospital extract.
presentCycle: (It should be number).
chemoScheduleName: (Only chemo name).
- Ex: 2nd cycle Epirubicin , Ifosfamide chemotherapy.
- Extract the presentCycle as 2. chemoName as Epirubicin, Ifosfamide.
chemoScheduleDate: (Extract from the content).

preChemoMedications: (If pre chemo medication or pre medication word matched)
- Give me as array.
- Each medication should have drugName, dosage, instructions.
- Ex: Inj. Epirubicin 85 mg IV in 100 ml NS over 15 minutes(D1-D2 only)
- Here drugName: Inj. Epirubicin, dosage: 85 mg, instructions: IV in 100 ml NS over 15 minutes [DAY 1 TO DAY 2 only]. D1 means Day 1.
- If no data omit this section

chemoMedications: (If only chemo medication & don't pre or post chemo medication include here.)
- Give me as array.
- Each medication should have drugName, dosage, instructions.
- Ex: Inj. Epirubicin 85 mg IV in 100 ml NS over 15 minutes(D1-D2 only)
- Here drugName: Inj. Epirubicin, dosage: 85 mg, instructions: IV in 100 ml NS over 15 minutes [DAY 1 TO DAY 2 only]. D1 means Day 1.
- If no data omit this section

postChemoMedications:
- Give me as array.
- From advice & discharge.
- Extract and name it as medications.
- Each medication should have drugName, dosage, morning-afternoon-night, noOfDays, instructions.
- Ex: Inj. Peg ANC count 6 mg - s/c ( 24 hours after chemotherapy )
- Here, drugName: Inj. Peg ANC count, dosage: 6 mg, morning: -, afternoon: -, night: -, instructions: s/c ( 24 hours after chemotherapy ).
- Ex2: Tab. Domstal 10 mg 1-0-1 before food for 5 days
- Here drugName: Tab. Domstal, dosage: 10 mg, morning: 1, afternoon: 0, night: 1, instructions: Before food, noOfDays: 5
- If no data omit this section

Note:
- All content's first letter should capital letter.
- If any value not there, means set empty string.

==============================
History
Rules:

From history or present illnes, all contents and their respective dates.
Each content group under the extracted date. If same date means combine the informations.
Don't duplicate information, if already attached anywhere omit here. For example if surgery info already included means omit here.

{
<date>: {
<action>: <content>
}
}

==============================
Blood Investigation
==============================

- Extract Blood investigation related or blood test related informations.
- Ca 125, CA125 -> if the word matched in History, extract the value and date. Under the date name: value should go.
- Ex: CA125 done on 26.09.2024-548.90.
- Here, name: CA125, date: 26 Sep, 2024, value: 548.90.
- Ex2: Hb-11.5 Here, name: Hb, value: 11.5.
- Like above example extract all blood test related informations.
- Value should be valid for name, otherwise omit that field.

{
  <date>: {
    <name>: <value>
  }
}

Rules:
- Each document group by document date
- Each content should comes under the corresponding date.
- If no date, then it should comes under the document date.
- If no content matched means, omit the section.

---------------------------------------------------

Ex:

{
  "documents": [
    {
      "documentName": "Name of the document",
      "documentDate": "DD MMM, YYYY",
      ...otherfields,
      "<date>": {
        ...content associated to the date
      }
    }
  ]
}
"extractedTables": [
        {
          "tableName": "Table 1 (or use the printed table heading if available)",
          "columns": ["Column 1 Name", "Column 2 Name", "Column 3 Name"],
          "rows": [
            {
              "Column 1 Name": "row 1 data",
              "Column 2 Name": "row 1 data",
              "Column 3 Name": "row 1 data"
            }
          ]
        }
      ],
      "parsedDocumentPercentage": 0
"parsedDocumentPercentage": any number
}

Note:
- Give me documents as array.
- Don't duplicate the content, if any content assigned in any other section means omit that.
"""

SYSTEM_PROMPT_PET_CT = """
PRE-PROCESSING (before extracting anything):
1. Strip grounding tags <|ref|>...<|/ref|>, <|det|>...<|/det|>, and bbox coords like [[693,342,925,353]].
2. Remove repeated page-header lines (patient name+date reprinted every page) — keep only first occurrence.
3. Skip pages with only image/title tags and no readable text.
4. Fix obvious OCR artefacts (split words, stray spaces, LaTeX sup like \\(4^{\\text{th}}\\) -> "4th") using context. NEVER alter numeric medical values (SUV, dose, size, glucose, creatinine, weight, mCi, MBq, mg/dl, kg, mm, cm).

All dates: DD MMM, YYYY (e.g. "22 Oct, 2025"). All JSON keys: camelCase.

Handles ANY PET/PET-CT/PET-MRI report: any centre/country, any radiotracer (18F-FDG, 68Ga-PSMA, 68Ga-DOTATATE, 18F-NaF, etc.), any body region, any format.

## SECTION 1 — METADATA
documentName: always "PET-CT Scan".
hospitalName: imaging centre name; null if absent.
hospitalId: patient ID at THIS facility (Patient ID/UHID/Id.No/MRN/Reg No) — first one found. NOT accessionNo.
accessionNo: Accession No if stated; else null.
documentDate: scan date, or report date if scan date absent. DD MMM, YYYY.
reportDate: signing date only if explicitly different from documentDate; else null.
referredBy: referring doctor, title+name only, strip ALL degrees/dept/institution. e.g. "DR.JEGAN NIWAS K MD DNB ONCOLOGY" -> "Dr. Jegan Niwas K".

## SECTION 2 — DEMOGRAPHICS
personalInfos: {
  patientName: full name as printed.
  age: exactly as written (e.g. "61Y", "29Y/F"); strip /F /M suffix into gender but keep original string here.
  gender: Male|Female|Other|null.
  dob: YYYY-MM-DD ONLY if explicitly printed; never calculate from age; null if absent (never "").
  patientWeight: kg if stated (often in technique table); null if absent.
  patientId: secondary ID if distinct from hospitalId; else null.
}

## SECTION 3 — SCAN TECHNIQUE
May appear as prose or key-value table — map all label variants.
scanTechnique: {
  studyType: modality/study name only, no acquisition mode, e.g. "FDG PETCT SCAN WHOLE BODY WITH CEMRI BRAIN", "PET/CT Whole Body" (label: Study/Study Type).
  radiotracer: normalised name, e.g. "18F-FDG", "68Ga-PSMA" (convert LaTeX {}^{18}F -> "18F"). Label: Radio tracer/Tracer/Radio-Isotope/Radiopharmaceutical.
  dose: activity+route both required, e.g. "5.9 mCi i.v"; if only route with no activity value -> null (never store partial). Label: Dose & Route/Administered Activity/Dose/Activity.
  uptakeTime: e.g. "60 min"; null if absent. Label: Uptake Time/Delay.
  ivContrast: contrast details if used, e.g. "Omnipaque 80 ml"; null if none/unstated.
  bloodGlucose: value+unit as written, e.g. "104 mg/dl". Label: Blood glucose/BGL.
  serumCreatinine: value+unit as written. Label: S.Creatinine/Sr. Creatinine.
  patientWeight: kg if in technique table — copy same value to personalInfos.patientWeight too.
  scanRegion: e.g. "Whole Body", "Skull Base to Mid-Thigh". Label: Extent of Study/Scan Region/Coverage.
  scanner: model/make if stated, e.g. "GE Discovery STE with HD PET-CT"; null if absent.
  acquisitionProtocol: extra protocol notes (e.g. "Motion Free acquisition", "SUVs Based on body weight") combined into one string; null if nothing beyond studyType.
}

## SECTION 4 — CLINICAL HISTORY
clinicalHistory: {
  indication: stated reason for scan + current symptoms not already in priorScans, e.g. "Staging", "c/o vomiting 1 week and fever".
  priorScans: [ one object per distinct prior event (PET CT/CT/MRI/Biopsy/HPE/Surgery/Chemotherapy/Radiotherapy/Mammogram/Ultrasound), chronological earliest-first:
    {
      scanDate: DD MMM, YYYY (convert "05.04.2022"->"05 Apr, 2022"; "April 2022"->"01 Apr, 2022").
      scanType: one of PET CT|CT|MRI|Biopsy|HPE|Surgery|Chemotherapy|Radiotherapy|Mammogram|Ultrasound, else as written.
        Surgery keywords: S/P, wide excision, underwent, operated. Chemo keywords: post chemo, cycles of chemo, adjuvant chemo. RT keywords: RT, radiotherapy, EBRT. HPE keywords: HPE, histopathology, biopsy result.
      findings: full verbatim result for this event (procedure+organ for Surgery; regimen/cycles/fractions for Chemo/RT; histological diagnosis for HPE).
      treatment: related treatment distinct from the event itself, e.g. "4 cycles of chemo after surgery"; null if none.
      response: explicitly stated response, e.g. "Good reduction", "disease progression"; null if none.
    }
  ]
  currentMedications: array of ongoing drugs, e.g. ["Letrozole","Denosumab"]; null if none mentioned.
  comparisonScan: date of formally-compared scan from "Compared with previous scan dated..."; DD MMM, YYYY; null if no formal comparison.
}

## SECTION 5 — FINDINGS BY REGION
Extract ALL findings under whatever anatomical headings the report uses (head and neck/brain/breasts & thorax/PETCT chest/abdomen/pelvis/skeletal system/soft tissues/general/etc).

findings: {
  "<regionInCamelCase>": {
    normalFindings: [ one string per statement of normal/unremarkable/physiological FDG distribution/no significant abnormality/no FDG avid lesions/no lymphadenopathy/no metabolically active disease — copy verbatim, never omit (medically significant). Physiological FDG in brain/heart/liver/spleen/kidneys/gut/bladder is ALWAYS normalFindings, never abnormal. ],
    abnormalFindings: [ one object per distinct lesion/node-group/effusion/structural change/incidental finding. Post-op changes = abnormal even if non-FDG avid. Incidental non-oncological findings (effusion, calcification, atelectasis, fibrosis, microliths, osteoporosis) = abnormal with fdgAvid false/null as appropriate. [] if none.
      {
        site: precise location, e.g. "left internal mammary lymph node".
        description: full verbatim finding text, no truncation.
        fdgAvid: true (uptake above background) | false (explicitly non-avid/cold) | null (not stated).
        suvMax: NUMBER (e.g. 3.0) OR exact string "Non FDG avid" if report uses that phrase — NEVER null/0 for that phrase; null only if truly not stated.
        suvMaxPrevious: previous SUV max stated inline, e.g. "(Previously SUV max 2.3)" -> NUMBER|"Non FDG avid"|null.
        size: as written, e.g. "10 x 8 mm"; null if absent.
        changeFromPrevious: Increased|Decreased|Stable|New|Resolved|Partial Resolution|Reappeared|null (infer from report language; null if cannot infer).
        morphology: structural descriptor, e.g. "sclerotic", "hypermetabolic nodule", "post-operative change"; null if none stated.
      }
    ]
  }
}

## SECTION 6 — SUV COMPARISON TABLE
If report has a multi-column table comparing SUV max across time-points (commonly skeletal lesions): scan dates as column headers (convert to DD MMM, YYYY), first column = sites, extract every row.
suvComparisonTable: [ { site: as written in col 1, suvValues: [ { scanDate, suvMax: NUMBER or "Non FDG avid" (never null/0 for that phrase) } ] } ]
Omit entirely if no such table.

## SECTION 7 — IMPRESSION
May appear twice (early summary + final formal IMPRESSION PETCT). ALWAYS use the LAST/most complete; discard earlier duplicate.
impressionComparedWith: date from "Compared with previous scan dated..." in impression section; DD MMM, YYYY; null if absent.
impression: [ split strictly — each top-level bullet (•,*,❖,leading dash) = one string; each sub-bullet ("- " indented) = its OWN separate string, never merged with parent; sub-sub-bullets ("o "/double-indent) also separate. Strip leading bullet chars. Never merge/split bullets. Preserve verbatim text. ]
overallAssessment: one of Complete Response|Partial Response|Stable Disease|Mixed Response|Progressive Disease|Disease Progression at Specific Sites|Reappearance of Disease|Post-Treatment Changes|No Evidence of Disease|Inconclusive|null.
  Guidance: new omental/peritoneal/distant mets not on prior scan -> Progressive Disease. Reappearance in post-op bed -> Reappearance of Disease. Some lesions up, others down -> Mixed Response. "Good response to therapy" + no new disease -> Partial Response. "No evidence of metabolically active disease" + all resolved -> No Evidence of Disease.

## SECTION 8 — SIGNATORIES
Every named doctor (primary verifying/signing, "in consultation with", named panel members). Exclude referring doctor (goes in referredBy).
signatories: [ { name (title+name as written), designation, degrees (as written, stripped from name, or null), registrationNo (e.g. "TNMC Reg No: 74817", or null), role: Primary (main signer)|Consultation (listed under "in consultation with")|Panel (named but didn't sign)|null } ]

## SECTION 9 — LESION SUMMARY (flat cross-region quick reference)
Every distinct lesion with at least one of fdgAvid/suvMax/size/changeFromPrevious, from findings + suvComparisonTable. Merge duplicates into one entry: prefer suvMax from comparison table, description/size from prose findings.
lesionSummary: [ { site, region (camelCase), fdgAvid, suvMaxCurrent: NUMBER|"Non FDG avid"|null, suvMaxPrevious: NUMBER|"Non FDG avid"|null, size, changeFromPrevious } ]

## OUTPUT STRUCTURE
{
  "documents": [{
    "documentName": "PET-CT Scan", "documentDate": "", "reportDate": null, "hospitalName": null,
    "hospitalId": "", "accessionNo": null, "referredBy": "",
    "personalInfos": {"patientName": "", "age": "", "gender": "", "dob": null, "patientWeight": null, "patientId": null},
    "scanTechnique": {"studyType": "", "radiotracer": "", "dose": null, "uptakeTime": null, "ivContrast": null,
      "bloodGlucose": null, "serumCreatinine": null, "patientWeight": null, "scanRegion": "", "scanner": null,
      "acquisitionProtocol": null},
    "clinicalHistory": {"indication": "", "priorScans": [], "currentMedications": null, "comparisonScan": null},
    "findings": {},
    "suvComparisonTable": [],
    "impressionComparedWith": null, "impression": [], "overallAssessment": null,
    "lesionSummary": [], "signatories": []
  }],
  "extractedTables": [{"tableName": "Table 1 or printed heading", "columns": ["Col1","Col2"], "rows": [{"Col1": "v", "Col2": "v"}]}],
  "parsedDocumentPercentage": 0
}

FINAL RULES:
1. NEVER alter any numeric value (SUV, dose, glucose, creatinine, weight, measurements, dates).
2. "Non FDG avid" is a VALID suvMax string — preserve exactly, never replace with null or 0.
3. Include ALL normal-organ statements in normalFindings — never omit.
4. Every lesion (prose OR table) must appear once in lesionSummary, no duplicates.
5. Repeated page headers: use only first occurrence, never duplicate from later pages.
6. Skip pure-image pages (no text).
7. Section appears multiple times (e.g. Impression): use most complete, discard shorter duplicates.
8. scanTechnique.dose must have BOTH activity value AND route; route-only -> null.
9. studyType = modality name only; acquisition mode details go in acquisitionProtocol.
10. hospitalId = Patient ID/UHID/MRN, NOT Accession No (that's accessionNo).
11. Each impression sub-bullet is its own array entry — never concatenate parent+child.
12. Capture every signatory (primary/consultation/panel) separately.
13. parsedDocumentPercentage: honest 0-100 estimate of completeness.
14. Output ONLY the JSON object — no preamble, no code fences.
"""

SYSTEM_PROMPT_CT_SCAN = """
PRE-PROCESSING (before extracting anything):
1. Strip grounding tags <|ref|>...<|/ref|>, <|det|>...<|/det|>, and bbox coords like [[693,342,925,353]].
2. Remove repeated page-header lines (patient name+date reprinted every page) — keep only first occurrence.
3. Skip pages with only image/title tags and no readable text.
4. Fix obvious OCR artefacts (split words, stray spaces, LaTeX sup like \\(cm^2\\) -> "cm2") using context. NEVER alter numeric medical values (measurements, HU, dates, contrast volumes, doses, slice thickness).

All dates: DD MMM, YYYY (e.g. "15 Jan, 2025"). All JSON keys: camelCase.

Handles ANY CT report: plain/contrast/CECT/HRCT/NCCT/angiography/urography/colonoscopy/myelogram/perfusion/dual-energy/low-dose/guided-biopsy/staging/follow-up, any body region, any format (prose/bullets/table).

## SECTION 1 — METADATA
documentName: always "CT Scan".
hospitalName: imaging centre/hospital name; null if absent.
hospitalId: patient ID at THIS facility (Patient ID/UHID/Id.No/MRN/Reg No) — first one found. NOT accessionNo.
accessionNo: Accession No if stated; else null.
documentDate: scan date (not report date) if they differ. DD MMM, YYYY.
reportDate: signing/issue date only if explicitly different from documentDate; else null.
referredBy: referring doctor, title+name only, strip degrees/dept. e.g. "Dr. Ramesh Kumar MD DM Oncology" -> "Dr. Ramesh Kumar".

## SECTION 2 — DEMOGRAPHICS
personalInfos: {
  patientName: full name as printed.
  age: exactly as written (e.g. "45Y", "38Y/F"), strip /F /M suffix into gender but keep original string here.
  gender: Male|Female|Other|null.
  dob: YYYY-MM-DD ONLY if explicitly printed; never calculate from age; null if absent (never "").
  patientId: secondary ID if distinct from hospitalId; else null.
}

## SECTION 3 — SCAN TECHNIQUE
May appear as prose, table, or "Technique"/"Protocol" section — map all label variants.
scanTechnique: {
  studyType: full study name only, no protocol details, e.g. "CECT Chest Abdomen Pelvis", "HRCT Chest", "NCCT Brain", "CT Guided Biopsy Liver".
  bodyRegion: region(s) scanned, comma-separated if multiple, e.g. "Chest Abdomen Pelvis".
  scanner: make/model if stated (labels: Scanner/Equipment/Machine/System); else null.
  sliceCount: e.g. "128 Slice"; null if absent.
  sliceThickness: e.g. "5 mm" (labels: Slice Thickness/Reconstruction/Section Thickness); null if absent.
  contrast: true if CECT/with contrast/+C/CE/post-contrast; false if plain/NCCT/non-contrast; null if undeterminable.
  contrastAgent: name+dose, e.g. "Omnipaque 300 80 ml IV"; null if plain or unnamed.
  contrastPhases: array as written, e.g. ["Arterial phase","Portal venous phase"]; [] if none.
  serumCreatinine: value+unit if stated pre-contrast; else null.
  oralContrast: true|false|null.
  scanCoverage: landmark range if stated, e.g. "Lung apex to inguinal ligament"; else null.
  kvp, mas, reconstructionAlgorithm, acquisitionProtocol: as stated; null if absent. acquisitionProtocol covers extras like "Dual energy CT", "ECG gating" — never append these into studyType.
}

## SECTION 4 — CLINICAL HISTORY
clinicalHistory: {
  indication: stated reason/clinical question + current symptoms not already in priorEvents, e.g. "Staging of known carcinoma right lung".
  priorEvents: [ one object per distinct prior event (CT/PET CT/MRI/Biopsy/HPE/Surgery/Chemotherapy/Radiotherapy/X-ray/Ultrasound/Mammogram), chronological earliest-first:
    {
      eventDate: DD MMM, YYYY (convert "12.03.2024"->"12 Mar, 2024"; "March 2024"->"01 Mar, 2024"; "2024"->"01 Jan, 2024").
      eventType: one of CT|PET CT|MRI|Biopsy|HPE|Surgery|Chemotherapy|Radiotherapy|X-ray|Ultrasound|Mammogram, else as written.
        Surgery keywords: S/P, resection, excision, -ectomy, operated, underwent. Chemo keywords: chemo(therapy), NACT, cycles of. RT keywords: RT, EBRT, radiation. HPE keywords: HPE, histopathology, biopsy result.
      findings: full verbatim result for this event (procedure+organ/side for Surgery; regimen/cycles/dose for Chemo/RT; histological diagnosis for HPE).
      treatment: related treatment distinct from the event itself; null if none.
      response: explicitly stated response assessment; null if none.
    }
  ]
  currentMedications: array of ongoing drugs at time of scan; null if none mentioned.
  comparisonStudy: { date: "DD MMM, YYYY", studyType: "CT" } from "Compared with CT dated..."; null if no formal comparison.
}

## SECTION 5 — FINDINGS BY REGION
Extract ALL findings, organised under whatever anatomical headings the report actually uses (brain/chest/lungs/liver/kidneys/bowel/lymph nodes/skeletal/vascular/etc — use report's own headings).

findings: {
  "<regionInCamelCase>": {
    normalFindings: [ one string per organ/structure stated as normal/unremarkable/no lesion/no lymphadenopathy/no effusion/no pneumothorax — copy verbatim, never omit, these are clinically significant ],
    abnormalFindings: [ one object per distinct lesion/mass/node-group/effusion/consolidation/structural change/incidental finding (cysts, haemangiomas, calcifications, post-op changes = abnormal too). If a lesion spans regions, place under primary region. [] if none.
      {
        site: precise location, e.g. "segment VI of liver", "right hilar lymph node".
        description: full verbatim finding text, no truncation.
        size: all axes as written, e.g. "3.2 x 2.8 x 2.1 cm"; null if absent.
        sizePrevious: inline comparison size if stated; else null.
        density: Hypodense|Isodense|Hyperdense|Heterogeneous|Fat density|Air density|Calcified|Mixed density|as written|null.
        huValue: e.g. "35 HU"; null if absent.
        enhancement: Homogeneous|Heterogeneous|Rim|Peripheral|Central|Nodular|Avid|Mild|Moderate|Marked|Non-enhancing|Arterial phase|Portal venous phase|Delayed|as written|null (null if no contrast given).
        morphology: shape/margin descriptors, e.g. "well-defined", "spiculated", "ground-glass opacity", "osteolytic"; null if none stated.
        lymphNodes: if this is a node finding: { shortAxis, longAxis, station, matted }; else null.
        calcification: true|false|null.
        changeFromPrevious: Increased|Decreased|Stable|New|Resolved|Partial Resolution|Reappeared|null (null if no prior study/cannot infer).
        impression: per-finding interpretation if stated here, e.g. "suspicious for malignancy"; else null.
      }
    ]
  }
}

## SECTION 6 — GLOBAL CT ASSESSMENTS (report-level, not per-lesion)
fluidCollections: { ascites: true|false|null, ascitesDescription: verbatim or null, pleuralEffusion: {right, left: true|false|null, description}, pericardialEffusion: true|false|null, otherCollections: [] }
pneumothorax: { present: true|false|null, side: Right|Left|Bilateral|null, description }
bonyFindings: { present: true|false|null, description (null if captured per-region instead) }
vascularFindings: { present: true|false|null, description (null if captured per-region instead) }

## SECTION 7 — MEASUREMENTS (flat list, all axes, from findings+impression)
measurements: [ { site, dimension (as written), dimensionPrevious (or null), unit, huValue (or null), changeFromPrevious } ]
Omit entirely if no measurements stated.

## SECTION 8 — LYMPH NODE SUMMARY (consolidated across all regions, reference not duplicate)
lymphNodeSummary: [ { station, region (camelCase findings key), shortAxis, longAxis, significant: true if enlarged/suspicious/FDG avid/necrotic else false/null, matted: true|false|null, changeFromPrevious } ]
Omit entirely if no nodes discussed.

## SECTION 9 — IMPRESSION
If impression appears twice (early Conclusion + final Impression), use the LAST/most complete; discard earlier duplicate.
impressionComparedWith: { date, studyType: "CT" } from impression section comparison; else null.
impression: [ split strictly — each top-level bullet = one string; each sub-bullet (indented "- ") = its OWN separate string, never merged with parent; sub-sub-bullets ("o " or double-indent) also separate. Strip leading bullet/number chars. Preserve verbatim text, no summarising. ]
overallAssessment: one of Complete Response|Partial Response|Stable Disease|Mixed Response|Progressive Disease|Disease Progression at Specific Sites|Reappearance of Disease|Post-Treatment Changes|No Evidence of Disease|Likely Benign|Likely Malignant|Indeterminate|Inconclusive|Normal Study|null.
  Guidance: new/enlarged lesions vs prior -> Progressive Disease. All lesions reduced -> Partial/Complete Response. No change -> Stable Disease. Mixed changes -> Mixed Response. Post-op only, no residual -> Post-Treatment Changes. Entirely normal -> Normal Study. No malignant evidence -> No Evidence of Disease. Benign features -> Likely Benign. Suspicious features -> Likely Malignant. Cannot characterise -> Indeterminate.

## SECTION 10 — SIGNATORIES
Every named doctor on the report (primary reporting/verifying radiologist, "in consultation with", named panel members). Exclude referring doctor (goes in referredBy).
signatories: [ { name (title+name as written), designation, degrees (or null), registrationNo (or null), role: Primary|Consultation|Panel|null } ]

## SECTION 11 — LESION SUMMARY (flat cross-region quick reference)
Every distinct lesion with at least one of size/enhancement/density/changeFromPrevious, pulled from findings+measurements. If same lesion in both, merge into one entry, prefer measurements[] value.
lesionSummary: [ { site, region (camelCase), size, sizePrevious, density, enhancement, calcification, changeFromPrevious, impressionNote (or null) } ]

## OUTPUT STRUCTURE
{
  "documents": [{
    "documentName": "CT Scan", "documentDate": "", "reportDate": null, "hospitalName": null,
    "hospitalId": "", "accessionNo": null, "referredBy": "",
    "personalInfos": {"patientName": "", "age": "", "gender": "", "dob": null, "patientId": null},
    "scanTechnique": {"studyType": "", "bodyRegion": "", "scanner": null, "sliceCount": null,
      "sliceThickness": null, "contrast": null, "contrastAgent": null, "contrastPhases": [],
      "serumCreatinine": null, "oralContrast": null, "scanCoverage": null, "kvp": null, "mas": null,
      "reconstructionAlgorithm": null, "acquisitionProtocol": null},
    "clinicalHistory": {"indication": "", "priorEvents": [], "currentMedications": null, "comparisonStudy": null},
    "findings": {},
    "fluidCollections": {"ascites": null, "ascitesDescription": null,
      "pleuralEffusion": {"right": null, "left": null, "description": null},
      "pericardialEffusion": null, "otherCollections": []},
    "pneumothorax": {"present": null, "side": null, "description": null},
    "bonyFindings": {"present": null, "description": null},
    "vascularFindings": {"present": null, "description": null},
    "measurements": [], "lymphNodeSummary": [],
    "impressionComparedWith": null, "impression": [], "overallAssessment": null,
    "lesionSummary": [], "signatories": []
  }],
  "extractedTables": [{"tableName": "Table 1 or printed heading", "columns": ["Col1","Col2"], "rows": [{"Col1": "v", "Col2": "v"}]}],
  "parsedDocumentPercentage": 0
}

FINAL RULES:
1. NEVER alter any numeric value (measurements/HU/contrast volumes/kVp/mAs/slice thickness).
2. Include ALL normal-organ statements in normalFindings — never omit.
3. Every lesion with size/density/enhancement data must appear once in lesionSummary, no duplicates.
4. Every lymph node finding must appear once in lymphNodeSummary, no duplicates.
5. Repeated page headers: use only first occurrence, never duplicate from later pages.
6. Skip pure-image pages (no text).
7. Duplicate Impression sections: use most complete, discard shorter earlier version.
8. Each impression sub-bullet is its own array entry — never concatenate parent+child.
9. hospitalId = Patient ID/UHID/MRN, NOT Accession No (that's accessionNo).
10. studyType = name only; protocol extras go in acquisitionProtocol.
11. contrast: true=CECT/with contrast/+C/CE/post-contrast; false=plain/NCCT/non-contrast; null=undeterminable.
12. fluidCollections/pneumothorax/bonyFindings/vascularFindings = GLOBAL statements only; per-region findings still go in findings{}.
13. lymphNodeSummary must consolidate ALL nodal findings from ALL regions.
14. Capture every signatory (primary/consultation/panel) separately.
15. parsedDocumentPercentage: honest 0-100 estimate of completeness.
16. Output ONLY the JSON object — no preamble, no code fences.
"""

SYSTEM_PROMPT_MRI = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                   PRE-PROCESSING  —  DO THIS BEFORE READING                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
 
The input is raw OCR output. Before extracting any content:
 
1. Strip every grounding tag pair:  <|ref|>…<|/ref|>  and  <|det|>…<|/det|>
2. Strip every bounding-box coordinate block, e.g. [[693, 342, 925, 353]]
3. Strip OCR section-type labels inside grounding tags
   (text, sub_title, title, table, image, etc.) — these are OCR artefacts.
4. Remove repeated page-header lines (patient name + date printed at the top
   of every page) — keep only the FIRST occurrence for patient demographics.
5. Pages that contain ONLY <|ref|>image<|/ref|> or <|ref|>title<|/ref|> tags
   with no readable text — SKIP entirely; they are scan image frames.
6. Fix obvious OCR artefacts (split words, stray spaces, merged characters,
   LaTeX-style superscripts like \\(T_2\\) → "T2", \\(cm^2\\) → "cm2") using
   surrounding context.  NEVER alter any numeric medical value (measurements,
   ADC values, signal intensities, dates, doses, contrast volumes).
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All dates: DD MMM, YYYY  (e.g. "15 Jan, 2025").  All JSON keys: camelCase.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
This prompt handles ANY MRI report regardless of:
  • Hospital, imaging centre, or country
  • Body region (brain, spine, breast, abdomen, pelvis, liver, prostate,
    MSK, cardiac, whole body, etc.)
  • MRI type (plain, contrast, functional, spectroscopy, MR angiography,
    MR urography, MRCP, fMRI, DWI, DCE-MRI, MR myelogram, etc.)
  • Report format (prose, bullet-point, tabular, mixed, sub-headed)
 
════════════════════════════════════════════════════════════════════════════════
SECTION 1 — DOCUMENT & FACILITY METADATA
════════════════════════════════════════════════════════════════════════════════
 
Extract:
  documentName     — always "MRI Scan".
  hospitalName     — name of the imaging centre / hospital as written.
                     null if not stated.
  hospitalId       — patient identifier used BY THIS FACILITY.
                     Check all of: Patient ID, UHID, MRN, Id.No, Reg No.
                     Use the FIRST one found.  Do NOT use Accession No here.
  accessionNo      — Accession No / Accession Number if stated; null otherwise.
  documentDate     — date the scan was performed.
                     If scan date and report date differ, use SCAN DATE here.
                     DD MMM, YYYY.
  reportDate       — date the report was signed/issued if explicitly different
                     from documentDate; null if same or not separately stated.
  referredBy       — referring / requesting doctor: title + full name only.
                     Strip ALL degrees (MD, MBBS, DNB, DM, MS, FRCS, etc.)
                     and department/institution names.
                     e.g. "Dr. Ramesh Kumar MD DM Neurology" → "Dr. Ramesh Kumar"
 
════════════════════════════════════════════════════════════════════════════════
SECTION 2 — PATIENT DEMOGRAPHICS
════════════════════════════════════════════════════════════════════════════════
 
personalInfos: {
  patientName,      — Full name as printed on the report.
  age,              — As written (e.g. "45Y", "38Y/F", "32 years").
                     Strip the sex suffix (/F, /M) — sex goes into gender.
  gender,           — Male | Female | Other | null.
  age,              — Extract the age string exactly as written on the report (e.g., "45Y", "38Y/F", "52 years"). Do not alter it.
  dob,              — YYYY-MM-DD. Extract ONLY if explicitly printed on the document. DO NOT calculate or infer the DOB from the age. If not explicitly stated, set to null. NEVER return "".
  patientId         — Secondary patient ID if distinct from hospitalId; null otherwise.
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 3 — SCAN TECHNIQUE
════════════════════════════════════════════════════════════════════════════════
 
Technique details may appear as a prose paragraph, a key-value table, or a
dedicated "Technique" / "Protocol" section.  Map all label variations to the
fields below.
 
scanTechnique: {
  studyType,         — Full MRI study name as written on the report header
                       or technique section.
                       e.g. "MRI Brain with Contrast", "MRI Whole Spine Plain",
                            "CECT MRI Abdomen and Pelvis", "MRI Breast Screening",
                            "MRI Brain with CE-MRI", "MRCP", "MR Angiography Brain",
                            "Diffusion Weighted MRI Brain", "MRI Prostate with DWI".
                       Use ONLY the study name — do not append protocol details.
 
  bodyRegion,        — Anatomical region(s) scanned.
                       e.g. "Brain", "Whole Spine", "Lumbar Spine",
                            "Abdomen and Pelvis", "Breast (Bilateral)",
                            "Prostate", "Liver", "Knee (Left)".
                       If multiple regions → comma-separated string.
 
  scanner,           — MRI machine make / model / tesla strength if stated.
                       e.g. "Siemens MAGNETOM Vida 3T",
                            "GE SIGNA Pioneer 3.0T",
                            "Philips Ingenia 1.5T".
                       If only field strength stated → include that.
                       null if not stated.
                       Table label: Scanner / Equipment / Machine / System.
 
  fieldStrength,     — Magnetic field strength as written.
                       e.g. "1.5 Tesla", "3.0 T", "3T", "1.5T".
                       Extract from scanner name if not separately stated.
                       null if not determinable.
 
  sequences,         — Array of MRI sequences performed.
                       Each element is the sequence name as written.
                       e.g. ["T1W", "T2W", "FLAIR", "DWI", "ADC", "SWI",
                             "STIR", "GRE", "T1W post-contrast", "DCE",
                             "MRS", "TOF MRA", "T2* GRE", "T2 SPAIR",
                             "DIXON", "VIBE", "HASTE", "MRCP"].
                       Extract from technique section and/or sequence headings
                       within the findings.  Deduplicate.
                       [] if none listed.
 
  plane,             — Imaging plane(s) used if stated.
                       e.g. "Axial, Sagittal, Coronal",
                            "Multiplanar reconstruction".
                       null if not stated.
 
  sliceThickness,    — Slice thickness if stated, e.g. "5 mm", "3 mm".
                       null if not stated.
 
  contrast,          — true if contrast was administered; false if plain/non-contrast;
                       null if not determinable.
 
  contrastAgent,     — Contrast agent name + dose/volume if stated.
                       e.g. "Gadovist 10 ml IV", "Gadopentetate dimeglumine 20 ml",
                            "Dotarem 0.1 mmol/kg IV".
                       null if plain study or agent not named.
                       Label variants: Contrast / IV Contrast / Gadolinium /
                                       Contrast agent / CE.
 
  contrastTiming,    — Contrast phase(s) if stated.
                       e.g. "Pre and post contrast", "Dynamic + delayed",
                            "Arterial, portal venous, delayed phases".
                       null if not stated.
 
  serumCreatinine,   — Value + unit if stated before contrast (renal safety check).
                       e.g. "0.9 mg/dl".  null if not stated.
 
  specialTechniques, — Any advanced / functional techniques used.
                       Array of strings; each element one technique.
                       e.g. ["DWI b=0,1000", "ADC mapping", "DCE perfusion",
                             "MR Spectroscopy (TE 144ms)", "MR Arthrography",
                             "Susceptibility Weighted Imaging", "Tractography"].
                       [] if none.
 
  acquisitionProtocol — Any other protocol details not captured above.
                        e.g. "Breath-hold technique", "Cardiac gating",
                             "Fat suppression used", "3D isotropic acquisition".
                        null if nothing additional.
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 4 — CLINICAL HISTORY & PRIOR EVENTS
════════════════════════════════════════════════════════════════════════════════
 
The clinical history / indication section often contains a chronological timeline
of prior imaging, biopsies, surgeries, and treatments.  Extract EVERY event.
 
clinicalHistory: {
 
  indication,        — The stated reason / clinical question for the current scan.
                       e.g. "Rule out intracranial metastasis",
                            "Follow-up post surgery", "Evaluation of breast lump",
                            "Back pain with radiculopathy".
                       Also include current presenting complaints / symptoms here.
                       This is the residual context NOT captured in priorEvents.
 
  priorEvents: [
    — One object per distinct prior event found in the clinical history section.
    — Covers: MRI, CT, PET-CT, Biopsy/HPE, Surgery, Chemotherapy, Radiotherapy,
              Mammogram, Ultrasound, X-ray.
    — Order chronologically (earliest first).
    {
      eventDate,     — DD MMM, YYYY.
                       Convert all formats:
                         "12.03.2024" → "12 Mar, 2024"
                         "March 2024" → "01 Mar, 2024"
                         "2024"       → "01 Jan, 2024"
      eventType,     — Classify as ONE of:
                         "MRI" | "CT" | "PET CT" | "Biopsy" | "HPE" |
                         "Surgery" | "Chemotherapy" | "Radiotherapy" |
                         "Mammogram" | "Ultrasound" | "X-ray" |
                         as written if none match.
                       Surgery keywords: S/P, wide excision, lumpectomy,
                         mastectomy, resection, operated, underwent.
                       Chemotherapy keywords: chemo, chemotherapy, NACT,
                         adjuvant chemo, cycles of.
                       Radiotherapy keywords: RT, EBRT, radiotherapy, radiation.
                       HPE keywords: HPE, histopathology, biopsy result, HPR.
      findings,      — Full finding / result statement for this event exactly
                       as written in the clinical history.
                       For Surgery: full procedure name and organ/side.
                       For Chemo/RT: regimen, cycles, dose/fractions if stated.
                       For HPE: histological diagnosis verbatim.
      treatment,     — Any treatment mentioned in context of this event but
                       distinct from the event itself; null if not stated.
      response       — Response assessment explicitly stated for this event;
                       null if not stated.
    }
  ],
 
  currentMedications,  — Drugs / treatments ONGOING at time of current scan.
                         Array of strings; null if none mentioned.
                         e.g. ["Tamoxifen", "Letrozole"]
 
  comparisonStudy      — Date and type of the study this MRI is formally compared
                         against (from "Compared with MRI dated …" or similar).
                         Format: { "date": "DD MMM, YYYY", "studyType": "MRI" }
                         null if no formal comparison stated.
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 5 — FINDINGS BY ANATOMICAL REGION
════════════════════════════════════════════════════════════════════════════════
 
Extract ALL findings without omission.  Organise by the anatomical section
headings exactly as they appear in the report.
 
Common headings (use whatever appears in the actual report):
  BRAIN | CEREBRUM | CEREBELLUM | BRAINSTEM | VENTRICLES | MENINGES |
  ORBITS | PARANASAL SINUSES | SKULL BASE | CALVARIUM |
  CERVICAL SPINE | THORACIC SPINE | LUMBAR SPINE | SACRUM | CORD |
  BREAST | AXILLA | CHEST | MEDIASTINUM |
  LIVER | GALLBLADDER | PANCREAS | SPLEEN | KIDNEYS | ADRENALS |
  BOWEL | PERITONEUM | RETROPERITONEUM |
  UTERUS | OVARIES | CERVIX | VAGINA | BLADDER | RECTUM | PROSTATE |
  LYMPH NODES | SKELETAL SYSTEM | SOFT TISSUES | VASCULAR | GENERAL
 
findings: {
  "<regionNameInCamelCase>": {
 
    normalFindings: [
      — Array of strings — one sentence per organ/structure/observation.
      — Include EVERY statement explicitly describing something as:
          • normal / appears normal / unremarkable / within normal limits
          • no abnormal signal intensity
          • no restricted diffusion
          • no abnormal enhancement
          • no evidence of focal lesion / mass / metastasis (when absence stated)
          • no significant lymphadenopathy (when absence stated)
      — Copy verbatim from the report; do not paraphrase.
      — Do NOT omit these — normal statements are clinically significant.
    ],
 
    abnormalFindings: [
      — Array of objects — one per DISTINCT lesion, mass, signal abnormality,
        structural change, enhancement pattern, or incidental finding.
      — If no abnormal findings in the region → empty array [].
      — Rules:
          • Post-operative changes, surgical defects = abnormal (structural).
          • Incidental findings (cysts, haemangiomas, degenerative changes,
            benign variants) = abnormal with appropriate morphology note.
          • Each distinct lesion/structure = one separate object.
      {
        site,                — Precise anatomical location.
                               e.g. "right temporal lobe",
                                    "L4-L5 intervertebral disc",
                                    "left breast upper outer quadrant",
                                    "right iliac lymph node".
        description,         — Full verbatim finding text for this lesion.
                               Do not truncate or paraphrase.
        size,                — Measurements as written.
                               e.g. "2.3 x 1.8 x 1.5 cm", "~12 mm", "3.4 cm".
                               For multi-dimensional: capture all axes.
                               null if not stated.
        sizePrevious,        — Size on comparison study if stated inline.
                               e.g. "Previously 2.8 cm".  null if not stated.
        t1Signal,            — T1 signal intensity relative to reference tissue.
                               "Hypointense" | "Isointense" | "Hyperintense" |
                               "Heterogeneous" | null.
        t2Signal,            — T2 signal intensity.
                               "Hypointense" | "Isointense" | "Hyperintense" |
                               "Heterogeneous" | null.
        flairSignal,         — FLAIR signal if stated.
                               "Hypointense" | "Isointense" | "Hyperintense" |
                               "Suppressed" | null.
        diffusionRestriction,— Diffusion restriction status for this lesion.
                               true  : restricted diffusion confirmed (high DWI,
                                       low ADC).
                               false : no restricted diffusion / facilitated
                                       diffusion.
                               null  : not stated or DWI not performed.
        adcValue,            — ADC value if stated for this lesion.
                               e.g. "0.8 x 10-3 mm2/s", "650 mm2/s".
                               null if not stated.
        enhancement,         — Enhancement pattern after contrast.
                               "Homogeneous" | "Heterogeneous" | "Rim" |
                               "Peripheral" | "Central" | "Nodular" |
                               "Avid" | "Mild" | "Moderate" | "Marked" |
                               "Non-enhancing" | "No enhancement" | null.
                               Use the exact descriptor from the report if it
                               does not match the above vocabulary.
                               null if contrast not given or not stated.
        morphology,          — Shape, margin, and structural descriptors.
                               e.g. "well-defined", "ill-defined",
                                    "lobulated", "spiculated",
                                    "cystic with solid component",
                                    "heterogeneous signal intensity",
                                    "ring-enhancing lesion",
                                    "disc herniation", "osteophyte",
                                    "cord compression".
                               null if no descriptor stated.
        changeFromPrevious,  — ONE of the following — infer from report language:
                               "Increased"          ← larger / increased signal
                               "Decreased"          ← smaller / reduced signal
                               "Stable"             ← no significant change
                               "New"                ← not seen on prior study
                               "Resolved"           ← completely gone
                               "Partial Resolution" ← smaller but still present
                               "Reappeared"         ← recurrence after resolution
                               null                 ← no prior study / cannot infer
        impression           — Interpretation stated specifically for this
                               finding if present in the findings section
                               (e.g. "likely meningioma", "consistent with
                               metastatic deposit", "favour benign aetiology").
                               null if no specific interpretation stated here
                               (full impression is captured separately).
      }
    ]
  }
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 6 — MRI-SPECIFIC GLOBAL ASSESSMENTS
════════════════════════════════════════════════════════════════════════════════
 
These are report-level summaries that apply ACROSS regions, not per-lesion.
 
diffusionRestriction: {
  — Global statement about diffusion restriction across the whole study.
  present,           — true | false | null.
  sites,             — Array of site strings where restriction is confirmed.
                       [] if none.
  adcValues,         — Array of { site, value } objects if ADC values stated.
                       [] if none.
  globalStatement    — Verbatim sentence(s) from the report about DWI findings
                       at the whole-study level.
                       null if not stated globally.
}
 
enhancement: {
  — Global statement about enhancement pattern across the whole study.
  present,           — true | false | null.
  pattern,           — Overall enhancement description if stated globally.
                       e.g. "Diffuse leptomeningeal enhancement",
                            "No abnormal parenchymal enhancement",
                            "Rim enhancement of the right temporal lesion".
                       null if no global statement.
  sites,             — Array of site strings where abnormal enhancement is
                       confirmed at the study level (supplement to per-lesion).
                       [] if none.
  globalStatement    — Verbatim sentence(s) from the report about enhancement
                       at the whole-study level.
                       null if not stated globally.
}
 
massEffect: {
  — Midline shift, herniation, hydrocephalus, cord compression summary.
  present,           — true | false | null.
  description        — Verbatim statement(s) about mass effect, shift, or
                       compression at the study level.
                       null if not stated.
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 7 — MEASUREMENTS SUMMARY
════════════════════════════════════════════════════════════════════════════════
 
Extract a flat indexed list of ALL measurements stated anywhere in the report
(findings AND impression).  One object per distinct measured structure.
 
measurements: [
  {
    site,            — Anatomical location of the measured structure.
    dimension,       — Full measurement as written (all axes).
                       e.g. "2.3 x 1.8 x 1.5 cm", "14 mm", "~3.4 cm".
    dimensionPrevious, — Previous measurement if stated for comparison; null otherwise.
    unit,            — "mm" | "cm" | "ml" | other as written.
    changeFromPrevious — "Increased" | "Decreased" | "Stable" | "New" |
                         "Resolved" | null.
  }
]
 
Omit measurements entirely if no measurements are stated in the report.
 
════════════════════════════════════════════════════════════════════════════════
SECTION 8 — IMPRESSION
════════════════════════════════════════════════════════════════════════════════
 
The impression / conclusion may appear:
  • Once at the end of the report (most common)
  • Both as an early summary/conclusion and as a final IMPRESSION section
  ALWAYS use the LAST / MOST COMPLETE occurrence.
  Discard earlier duplicates.
 
impressionComparedWith — Date and type of comparison study from the impression
                         section (e.g. "Compared with MRI dated 12 Jan, 2025").
                         Format: { "date": "DD MMM, YYYY", "studyType": "MRI" }
                         null if no comparison stated in impression.
 
impression: [
  — Strict bullet-splitting rules:
    1. Each TOP-LEVEL bullet (•, *, ❖, numbered item, or leading dash at the
       start of a paragraph) = one string.
    2. Each SUB-BULLET (lines starting with "- " indented under a parent) =
       a SEPARATE string.  Never concatenate parent + child bullets.
    3. Sub-sub-bullets (lines starting with "o " or double-indented) =
       also SEPARATE strings.
    4. Strip leading bullet/number characters from every string.
    5. Never merge two bullet points; never split one across two strings.
    6. Preserve full verbatim text — do not summarise.
]
 
overallAssessment — Infer ONE value from the impression language:
  "Complete Response"
  "Partial Response"
  "Stable Disease"
  "Mixed Response"
  "Progressive Disease"
  "Disease Progression at Specific Sites"
  "Reappearance of Disease"
  "Post-Treatment Changes"
  "No Evidence of Disease"
  "Likely Benign"
  "Likely Malignant"
  "Indeterminate"
  "Inconclusive"
  null
 
  Decision guidance:
    New lesions not present on prior study → "Progressive Disease"
    All known lesions reduced → "Partial Response" or "Complete Response"
    No change overall → "Stable Disease"
    Some increased, some decreased → "Mixed Response"
    Post-op scan, only surgical changes, no residual disease → "Post-Treatment Changes"
    No lesion found, study normal → "No Evidence of Disease"
    Single lesion with benign features, no malignancy suspected → "Likely Benign"
    Features suspicious for malignancy → "Likely Malignant"
    Cannot characterise → "Indeterminate"
 
════════════════════════════════════════════════════════════════════════════════
SECTION 9 — SIGNATORIES
════════════════════════════════════════════════════════════════════════════════
 
Extract EVERY named doctor associated with the report — including:
  • The primary reporting / verifying radiologist
  • Any "In consultation with" / second opinion doctor
  • Any panel members explicitly named with a designation
 
Do NOT include the referring doctor here (that goes in referredBy).
 
signatories: [
  {
    name,               — Title + full name as written.
                          e.g. "Dr. Anitha R.", "Dr. Suresh Babu".
    designation,        — Role/title as written.
                          e.g. "Consultant Radiologist",
                               "Senior Radiologist",
                               "Neuroradiologist".
    degrees,            — Qualifications string as written, stripped from name.
                          e.g. "MD, DMRD", "FRCR", "DNB Radiodiagnosis".
                          null if not stated.
    registrationNo,     — Medical registration number if stated; null otherwise.
    role                — "Primary" | "Consultation" | "Panel" | null.
  }
]
 
════════════════════════════════════════════════════════════════════════════════
SECTION 10 — LESION SUMMARY  (flat cross-region quick reference)
════════════════════════════════════════════════════════════════════════════════
 
Create a flat list of EVERY distinct lesion / abnormal finding that has at least
one of: size, diffusionRestriction, enhancement, or changeFromPrevious.
 
If the same lesion appears in both findings and measurements, merge into ONE
entry — prefer the measurement value from the measurements array.
 
lesionSummary: [
  {
    site,                  — Same text as in findings.abnormalFindings.
    region,                — camelCase key matching the findings region object.
    size,                  — As written; null if not stated.
    sizePrevious,          — null if not stated.
    t2Signal,              — As per findings; null if not stated.
    diffusionRestriction,  — true | false | null.
    enhancement,           — Enhancement descriptor; null if not stated.
    changeFromPrevious,    — Same vocabulary as abnormalFindings.changeFromPrevious.
    impressionNote         — Per-lesion interpretation from findings.impression
                             if present; null otherwise.
  }
]
 
════════════════════════════════════════════════════════════════════════════════
OUTPUT JSON STRUCTURE
════════════════════════════════════════════════════════════════════════════════
 
{
  "documents": [
    {
      "documentName": "MRI Scan",
      "documentDate": "DD MMM, YYYY",
      "reportDate": null,
      "hospitalName": null,
      "hospitalId": "",
      "accessionNo": null,
      "referredBy": "",
 
      "personalInfos": {
        "patientName": "",
        "age": "",
        "gender": "",
        "dob": "YYYY-MM-DD",
        "patientId": null
      },
 
      "scanTechnique": {
        "studyType": "",
        "bodyRegion": "",
        "scanner": null,
        "fieldStrength": null,
        "sequences": [],
        "plane": null,
        "sliceThickness": null,
        "contrast": null,
        "contrastAgent": null,
        "contrastTiming": null,
        "serumCreatinine": null,
        "specialTechniques": [],
        "acquisitionProtocol": null
      },
 
      "clinicalHistory": {
        "indication": "",
        "priorEvents": [],
        "currentMedications": null,
        "comparisonStudy": null
      },
 
      "findings": {},
 
      "diffusionRestriction": {
        "present": null,
        "sites": [],
        "adcValues": [],
        "globalStatement": null
      },
 
      "enhancement": {
        "present": null,
        "pattern": null,
        "sites": [],
        "globalStatement": null
      },
 
      "massEffect": {
        "present": null,
        "description": null
      },
 
      "measurements": [],
 
      "impressionComparedWith": null,
      "impression": [],
      "overallAssessment": null,
 
      "lesionSummary": [],
 
      "signatories": []
    }
  ],
  "extractedTables": [
        {
          "tableName": "Table 1 (or use the printed table heading if available)",
          "columns": ["Column 1 Name", "Column 2 Name", "Column 3 Name"],
          "rows": [
            {
              "Column 1 Name": "row 1 data",
              "Column 2 Name": "row 1 data",
              "Column 3 Name": "row 1 data"
            }
          ]
        }
      ],
      "parsedDocumentPercentage": 0
  "parsedDocumentPercentage": 0
}
 
════════════════════════════════════════════════════════════════════════════════
FINAL RULES
════════════════════════════════════════════════════════════════════════════════
 
1.  NEVER alter any numeric value: measurements, ADC values, signal intensities,
    contrast volumes, dates, or any other number.
2.  Include ALL normal organ/structure statements in normalFindings — never omit.
3.  Every distinct lesion with size, diffusion, or enhancement data must appear
    in lesionSummary with no duplicates.
4.  Repeated page headers (name + date) on every OCR page — use only the FIRST
    occurrence; never create duplicate data from subsequent page headers.
5.  Pages detected as pure images (only image/title tags, no text) — skip.
6.  If Impression appears multiple times, use the MOST COMPLETE version and
    discard shorter duplicates.
7.  Each impression sub-bullet = its own array entry.  Never concatenate
    parent and child bullets into one string.
8.  hospitalId = Patient ID / UHID / MRN — NOT Accession No.
    Accession No goes in accessionNo field.
9.  scanTechnique.studyType = study name only.  Protocol details go in
    acquisitionProtocol — never concatenate them into studyType.
10. scanTechnique.contrast:
      true  → contrast was given (keywords: with contrast, post-contrast, CEMRI,
               gadolinium, +C, W/C, CE).
      false → plain / non-contrast / without contrast.
      null  → not determinable from the report.
11. diffusionRestriction and enhancement objects at the root level capture
    GLOBAL report-level statements.  Per-lesion data still goes inside each
    abnormalFindings object in findings.
12. All signatories (primary, consultation, panel) must be captured.
13. parsedDocumentPercentage: your honest estimate (0-100) of how completely
    the full document content was captured.
14. Output ONLY the JSON object — no explanation, preamble, or code fences."""

SYSTEM_PROMPT_MAMMOGRAM = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                   PRE-PROCESSING  —  DO THIS BEFORE READING                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
 
The input is raw OCR output. Before extracting any content:
 
1. Strip every grounding tag pair:  <|ref|>…<|/ref|>  and  <|det|>…<|/det|>
2. Strip every bounding-box coordinate block, e.g. [[693, 342, 925, 353]]
3. Strip OCR section-type labels inside grounding tags
   (text, sub_title, title, table, image, etc.) — these are OCR artefacts.
4. Remove repeated page-header lines (patient name + date printed at the top
   of every page) — keep only the FIRST occurrence for patient demographics.
5. Pages that contain ONLY <|ref|>image<|/ref|> or <|ref|>title<|/ref|> tags
   with no readable text — SKIP entirely; they are mammogram image frames.
6. Fix obvious OCR artefacts (split words, stray spaces, merged characters)
   using surrounding context.  NEVER alter any numeric medical value
   (measurements, BI-RADS scores, density scores, dates, exposure values).
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All dates: DD MMM, YYYY  (e.g. "15 Jan, 2025").  All JSON keys: camelCase.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
This prompt handles ANY mammogram / mammography report regardless of:
  • Hospital, imaging centre, or country
  • Mammogram type (screening, diagnostic, digital, 3D tomosynthesis,
    contrast-enhanced, MBI, spot compression, magnification views)
  • Report format (prose, bullet-point, tabular, mixed, BI-RADS structured)
 
════════════════════════════════════════════════════════════════════════════════
SECTION 1 — DOCUMENT & FACILITY METADATA
════════════════════════════════════════════════════════════════════════════════
 
Extract:
  documentName     — always "Mammogram".
  labName          — Full name of the diagnostic lab / imaging centre / hospital
                     as written on the report.  null if not stated.
  labAddress       — Full address of the lab/centre including city, state,
                     pincode/zip if stated.  null if not stated.
  labContact       — Phone number(s) and/or email of the lab/centre if stated.
                     null if not stated.
  labAccreditation — Any accreditation / certification details stated
                     (e.g. "NABL accredited", "ACR accredited", "ISO 15189").
                     null if not stated.
  hospitalId       — Patient identifier used BY THIS FACILITY.
                     Check all of: Patient ID, UHID, MRN, Reg No, Id.No.
                     Use the FIRST one found.  Do NOT use Accession No here.
  accessionNo      — Accession No / Accession Number if stated; null otherwise.
  documentDate     — Date the mammography procedure was performed.
                     If scan date and report date differ, use SCAN DATE here.
                     DD MMM, YYYY.
  reportDate       — Date the report was signed/issued if explicitly different
                     from documentDate; null if same or not separately stated.
  reportTime       — Time the report was issued if stated; null otherwise.
                     e.g. "14:35", "2:35 PM".
  qrCode           — true if a QR code is mentioned or detected; false/null otherwise.
  barcode          — Barcode / accession barcode value if readable from OCR;
                     null otherwise.
 
════════════════════════════════════════════════════════════════════════════════
SECTION 2 — PATIENT DEMOGRAPHICS & PERSONAL INFORMATION
════════════════════════════════════════════════════════════════════════════════
 
personalInfos: {
  patientName,        — Full name as printed on the report.
  age,                — As written (e.g. "45Y", "38Y/F", "52 years").
                        Strip sex suffix (/F, /M) — sex goes into gender.
  gender,             — Male | Female | Other | null.
                        (Male mammograms exist — do not assume Female.)
  age,              — Extract the age string exactly as written on the report (e.g., "45Y", "38Y/F", "52 years"). Do not alter it.
  dob,              — YYYY-MM-DD. Extract ONLY if explicitly printed on the document. DO NOT calculate or infer the DOB from the age. If not explicitly stated, set to null. NEVER return "".
  patientId,          — Secondary patient ID if distinct from hospitalId; null.
  contactNumber,      — Patient phone number(s) if stated; null otherwise.
  email,              — Patient email if stated; null otherwise.
  address: {
    addressLine1,     — Door number, building, street, road.
    addressLine2,     — Area, locality, colony, nagar, landmark.
    city,             — City name.
    district,         — District name if stated; null otherwise.
    state,            — Full state name (expand abbreviations:
                        TN→Tamil Nadu, KA→Karnataka, MH→Maharashtra, etc.).
    country,          — Default "India" for Indian addresses; null if unclear.
    pincode           — 6-digit numeric code; null if not present.
  },
  insuranceDetails,   — Insurance provider, policy number, or any insurance
                        information if stated; null if not stated.
  emergencyContact: {
    name,             — Emergency contact name; null if not stated.
    relationship,     — Relationship to patient; null if not stated.
    contactNumber     — Emergency contact phone; null if not stated.
  }
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 3 — REFERRING DOCTOR INFORMATION
════════════════════════════════════════════════════════════════════════════════
 
referredBy: {
  name,               — Title + full name only.
                        Strip ALL degrees (MD, MBBS, DNB, DM, MS, etc.)
                        and institution/department names.
                        e.g. "Dr. Priya Sharma MS General Surgery" → "Dr. Priya Sharma"
  contactNumber,      — Referring doctor's phone if stated; null otherwise.
  email,              — Referring doctor's email if stated; null otherwise.
  specialty,          — Medical specialty as written; null if not stated.
                        e.g. "Oncology", "General Surgery", "Gynaecology".
  licenseNumber,      — Medical license / registration number if stated;
                        null otherwise.
  referralDate,       — Date of referral in DD MMM, YYYY; null if not stated.
  reasonForReferral,  — Clinical reason for requesting the mammogram as stated
                        by the referring doctor; null if not stated.
  specialInstructions,— Any specific instructions or concerns from the referring
                        doctor; null if not stated.
  preferredResultMode — Preferred communication method for results if stated
                        (e.g. "Email", "WhatsApp", "Hard copy"); null otherwise.
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 4 — TECHNICAL INFORMATION
════════════════════════════════════════════════════════════════════════════════
 
technicalDetails: {
  mammographyType,    — Type of mammography performed.
                        e.g. "Digital Mammography", "3D Tomosynthesis",
                             "Contrast-Enhanced Spectral Mammography (CESM)",
                             "Molecular Breast Imaging (MBI)",
                             "Screening Mammography", "Diagnostic Mammography".
                        As written if different from above.
 
  machine,            — Machine make / model / brand if stated.
                        e.g. "Hologic Selenia Dimensions", "GE Senographe Essential",
                             "Siemens Mammomat Revelation".
                        null if not stated.
                        Label variants: Equipment / Machine / System / Unit.
 
  views,              — Array of mammographic views performed.
                        Each element = one view as written.
                        e.g. ["CC right", "MLO right", "CC left", "MLO left",
                              "Spot compression right UOQ", "Magnification view",
                              "XCCL left", "ML right"].
                        [] if not listed.
 
  exposureSettings,   — Exposure / technical parameters if stated.
                        e.g. "kVp: 28, mAs: 80, target/filter: Mo/Mo",
                             "AGC mode, 26 kV, Rh/Rh filter".
                        null if not stated.
 
  compressionForce,   — Compression force in daN or N if stated; null otherwise.
                        e.g. "12 daN".
 
  technologistName,   — Name of the radiologic technologist / radiographer who
                        performed the procedure if stated; null otherwise.
 
  technologistCredentials, — Credentials/license of the technologist if stated;
                             null otherwise.
 
  procedureDateTime,  — Date and time of the mammography procedure if both stated.
                        Format: "DD MMM, YYYY HH:MM".
                        If only date → use documentDate.  null if neither.
 
  technicalNotes      — Any technical challenges, deviations from standard
                        protocol, repeat exposures, patient cooperation issues,
                        or quality notes; null if none stated.
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 5 — CLINICAL INDICATIONS & PATIENT HISTORY
════════════════════════════════════════════════════════════════════════════════
 
clinicalHistory: {
 
  mammogramType,      — "Screening" | "Diagnostic" | null.
                        Screening = no specific symptoms, routine check.
                        Diagnostic = specific symptom or abnormality being evaluated.
 
  indication,         — Primary reason for this mammogram as stated in the report.
                        e.g. "Routine screening", "Palpable lump right breast",
                             "Follow-up post lumpectomy", "Nipple discharge".
 
  patientSymptoms,    — Array of self-reported symptoms or concerns.
                        e.g. ["Palpable lump in right breast upper outer quadrant",
                              "Nipple discharge left breast",
                              "Skin changes over left breast"].
                        [] if no symptoms stated.
 
  familyHistory: {
    breastCancer,     — true | false | null.
    details           — Verbatim family history details if stated
                        (e.g. "Mother with breast cancer at age 48",
                              "Sister - BRCA1 positive").
                        null if not stated.
  },
 
  personalBreastHistory: {
    previousMammogram,    — true | false | null.
    lastMammogramDate,    — DD MMM, YYYY if stated; null otherwise.
    lastMammogramResult,  — Summary of previous mammogram result if stated;
                            null otherwise.
    previousBiopsy,       — true | false | null.
    biopsyResult,         — Histology result if stated; null otherwise.
    previousSurgery,      — true | false | null.
    surgeryDetails,       — Surgery name, side, date if stated; null otherwise.
                            e.g. "Left lumpectomy, Jan 2022",
                                 "Right mastectomy with reconstruction, 2019".
    implants,             — true | false | null.
    implantDetails        — Side, type if stated; null otherwise.
  },
 
  hormonalHistory: {
    hrtUse,               — true | false | null (hormone replacement therapy).
    oralContraceptives,   — true | false | null.
    lactating,            — true | false | null.
    pregnant,             — true | false | null.
    menopausalStatus      — "Pre-menopausal" | "Peri-menopausal" |
                            "Post-menopausal" | null.
  },
 
  otherClinicalFactors    — Any other relevant clinical factors mentioned;
                            null if none.
 
  priorEvents: [
    — One object per prior imaging, biopsy, surgery, or treatment event
      mentioned in the clinical history section.
    — Order chronologically (earliest first).
    {
      eventDate,          — DD MMM, YYYY.  Month-only → "01 MMM, YYYY".
      eventType,          — "Mammogram" | "Ultrasound" | "MRI Breast" |
                            "Biopsy" | "HPE" | "Surgery" | "Chemotherapy" |
                            "Radiotherapy" | "CT" | "PET CT" | as written.
      side,               — "Right" | "Left" | "Bilateral" | null.
      findings,           — Result / finding / diagnosis verbatim; null if not stated.
      treatment,          — Any treatment in context of this event; null if not stated.
      response            — Response / outcome if stated; null if not stated.
    }
  ]
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 6 — BREAST CHARACTERISTICS
════════════════════════════════════════════════════════════════════════════════
 
breastCharacteristics: {
  rightBreast: {
    density,            — ACR density category as written.
                          "Almost entirely fatty (A)" |
                          "Scattered areas of fibroglandular density (B)" |
                          "Heterogeneously dense (C)" |
                          "Extremely dense (D)" |
                          or the exact text if different.
    densityCategory,    — "A" | "B" | "C" | "D" | null (letter only).
    skinThickness,      — Skin thickness if measured; null otherwise.
    nipplePosition,     — Nipple position note if stated; null otherwise.
    recentSurgeryChanges — Any post-surgical changes noted; null if none.
  },
  leftBreast: {
    density,
    densityCategory,
    skinThickness,
    nipplePosition,
    recentSurgeryChanges
  },
  overallDensityStatement — Verbatim density statement from the report
                            if given as a combined bilateral statement;
                            null if sides stated separately.
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 7 — IMAGE FINDINGS
════════════════════════════════════════════════════════════════════════════════
 
findings: {
 
  rightBreast: {
    normalFindings: [
      — Array of verbatim statements describing normal tissue or absence of
        abnormality in the right breast.
        e.g. "No suspicious mass lesion identified.",
             "No microcalcifications noted.",
             "Skin and nipple appear normal."
    ],
    abnormalFindings: [
      — One object per distinct finding (mass, calcification cluster, asymmetry,
        architectural distortion, skin change, lymph node, etc.)
      {
        findingType,        — "Mass" | "Calcification" | "Asymmetry" |
                              "Architectural Distortion" | "Associated Feature" |
                              "Lymph Node" | "Skin Change" | "Nipple Change" |
                              "Intramammary Finding" | as written.
        location: {
          quadrant,         — "Upper Outer Quadrant (UOQ)" |
                              "Upper Inner Quadrant (UIQ)" |
                              "Lower Outer Quadrant (LOQ)" |
                              "Lower Inner Quadrant (LIQ)" |
                              "Central / Subareolar" | "Axillary Tail" |
                              as written.
          clockPosition,    — Clock-face position if stated; e.g. "12 o'clock".
          depthFromNipple,  — Distance from nipple if stated; e.g. "3 cm from nipple".
          depthInBreast     — "Anterior third" | "Middle third" |
                              "Posterior third" | as written | null.
        },
        size,               — Measurement as written; e.g. "12 x 8 mm", "~1.5 cm".
                              null if not stated.
        sizePrevious,       — Size on prior mammogram if stated inline; null otherwise.
        shape,              — "Round" | "Oval" | "Irregular" | as written | null.
        margin,             — "Circumscribed" | "Microlobulated" | "Obscured" |
                              "Indistinct" | "Spiculated" | as written | null.
        density_echogenicity,— Mass density: "High density" | "Equal density" |
                               "Low density" | "Fat-containing" | as written | null.
        calcifications: {
          present,          — true | false | null.
          morphology,       — "Typically benign" | "Suspicious" | "Amorphous" |
                              "Coarse heterogeneous" | "Fine pleomorphic" |
                              "Fine linear / branching" | as written | null.
          distribution,     — "Diffuse" | "Regional" | "Grouped / Clustered" |
                              "Linear" | "Segmental" | as written | null.
          number             — Approximate count if stated; null otherwise.
        },
        associatedFeatures, — Array of associated features stated.
                              e.g. ["Skin retraction", "Nipple retraction",
                                    "Skin thickening", "Trabecular thickening",
                                    "Axillary adenopathy", "Architectural distortion"].
                              [] if none stated.
        description,        — Full verbatim description of this finding.
        changeFromPrevious, — "Increased" | "Decreased" | "Stable" | "New" |
                              "Resolved" | "Partial Resolution" | null.
        impression          — Radiologist's interpretation of this specific
                              finding if stated in findings section
                              (e.g. "likely benign", "suspicious for malignancy",
                               "indeterminate"); null if not stated here.
      }
    ]
  },
 
  leftBreast: {
    normalFindings: [],
    abnormalFindings: []
    — Same structure as rightBreast above.
  },
 
  axilla: {
    right: {
      normalFindings: [],
      abnormalFindings: []
    },
    left: {
      normalFindings: [],
      abnormalFindings: []
    }
  },
 
  comparisonWithPrevious   — Verbatim statement about comparison with prior
                             mammogram(s) if a general comparison statement
                             is made beyond per-finding comparisons.
                             null if no prior mammogram or no general comparison.
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 8 — BI-RADS ASSESSMENT
════════════════════════════════════════════════════════════════════════════════
 
birads: {
  rightBreast: {
    category,           — Numeric category: 0 | 1 | 2 | 3 | 4 | 4A | 4B | 4C | 5 | 6
                          Extract from "BI-RADS 3", "BIRADS 4A", "Category 2", etc.
                          null if not stated.
    categoryDescription,— Standard description for the assigned category:
                          0  → "Incomplete – Need additional imaging"
                          1  → "Negative"
                          2  → "Benign"
                          3  → "Probably benign"
                          4  → "Suspicious"
                          4A → "Low suspicion for malignancy"
                          4B → "Moderate suspicion for malignancy"
                          4C → "High suspicion for malignancy"
                          5  → "Highly suggestive of malignancy"
                          6  → "Known biopsy-proven malignancy"
                          null if category is null.
    malignancyRisk,     — Stated or implied malignancy risk percentage if given;
                          null otherwise.  e.g. "2%", ">95%".
  },
  leftBreast: {
    category,
    categoryDescription,
    malignancyRisk
  },
  overall: {
    category,           — Overall combined BI-RADS if a single score is given
                          for both breasts together.
    categoryDescription,
    malignancyRisk
  }
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 9 — IMPRESSION & RECOMMENDATIONS
════════════════════════════════════════════════════════════════════════════════
 
The impression / conclusion may appear:
  • Once at the end of the report
  • Both as an early "Conclusion" and a final "Impression" section
  ALWAYS use the LAST / MOST COMPLETE occurrence; discard shorter duplicates.
 
impressionComparedWith  — Date and type of prior study this report is formally
                          compared against; null if no comparison stated.
                          Format: { "date": "DD MMM, YYYY", "studyType": "Mammogram" }
 
impression: [
  — Strict bullet-splitting rules:
    1. Each TOP-LEVEL bullet (•, *, ❖, numbered item, or leading paragraph) =
       one string.
    2. Each SUB-BULLET (lines starting with "- " under a parent) = a SEPARATE
       string.  Never concatenate parent + child bullets.
    3. Strip leading bullet/number characters from every string.
    4. Preserve full verbatim text — do not summarise.
]
 
recommendations: [
  — Array of strings — one per distinct recommendation.
  — Extract from both the impression section and any separate
    "Recommendation" / "Advice" / "Suggested" section.
  — e.g. "Ultrasound-guided core needle biopsy of right breast mass recommended.",
          "Follow-up mammogram in 6 months.",
          "Clinical correlation advised.",
          "MRI breast for further characterisation.",
          "Routine annual screening mammogram recommended."
  — [] if no recommendations stated.
]
 
uncertaintiesAndLimitations — Any stated limitations of the examination,
                               uncertainties in interpretation, or technical
                               caveats.  null if none stated.
                               e.g. "Dense breast tissue limits sensitivity.",
                                    "Motion artefact noted on CC left view."
 
overallAssessment — Infer ONE value from impression language:
  "Normal / Negative"               ← BI-RADS 1
  "Benign Finding"                  ← BI-RADS 2
  "Probably Benign"                 ← BI-RADS 3, short-term follow-up
  "Suspicious"                      ← BI-RADS 4 (any sub-category)
  "Highly Suggestive of Malignancy" ← BI-RADS 5
  "Known Malignancy"                ← BI-RADS 6
  "Incomplete – Further Imaging Needed" ← BI-RADS 0
  "Inconclusive"
  null
 
════════════════════════════════════════════════════════════════════════════════
SECTION 10 — SIGNATORIES
════════════════════════════════════════════════════════════════════════════════
 
Extract EVERY named doctor / professional associated with the report.
 
signatories: [
  {
    name,               — Title + full name as written.
                          e.g. "Dr. Anitha R.", "Dr. Suresh Babu".
    designation,        — Role/title as written.
                          e.g. "Consultant Radiologist",
                               "Breast Imaging Specialist",
                               "Senior Radiologist".
    degrees,            — Qualifications string as written, stripped from name.
                          e.g. "MD, DMRD", "FRCR", "DNB Radiodiagnosis".
                          null if not stated.
    registrationNo,     — Medical registration number if stated; null otherwise.
                          e.g. "TNMC Reg No: 74817", "MCI No: 12345".
    interpretationDate, — Date radiologist interpreted / signed, DD MMM, YYYY.
                          null if same as reportDate or not stated.
    interpretationTime, — Time of interpretation if stated; null otherwise.
    role                — "Primary" | "Consultation" | "Panel" | null.
  }
]
 
════════════════════════════════════════════════════════════════════════════════
SECTION 11 — LESION SUMMARY  (flat cross-breast quick reference)
════════════════════════════════════════════════════════════════════════════════
 
Create a flat list of EVERY distinct abnormal finding across both breasts and
axillae that has at least one of: size, BI-RADS linkage, calcifications, or
changeFromPrevious.
 
lesionSummary: [
  {
    side,               — "Right" | "Left" | "Bilateral".
    region,             — "Breast" | "Axilla".
    findingType,        — As per findings section.
    location,           — Quadrant or clock position as written.
    size,               — As written; null if not stated.
    sizePrevious,       — null if not stated.
    calcifications,     — true | false | null.
    changeFromPrevious, — Same vocabulary as findings.changeFromPrevious.
    impressionNote,     — Per-finding interpretation if stated; null otherwise.
    biradsCat           — BI-RADS category for this finding if individually
                          assigned; null if only overall score given.
  }
]
 
════════════════════════════════════════════════════════════════════════════════
OUTPUT JSON STRUCTURE
════════════════════════════════════════════════════════════════════════════════
 
{
  "documents": [
    {
      "documentName": "Mammogram",
      "documentDate": "DD MMM, YYYY",
      "reportDate": null,
      "reportTime": null,
      "labName": null,
      "labAddress": null,
      "labContact": null,
      "labAccreditation": null,
      "hospitalId": "",
      "accessionNo": null,
      "qrCode": null,
      "barcode": null,
 
      "personalInfos": {
        "patientName": "",
        "age": "",
        "gender": "",
        "dob": "YYYY-MM-DD",
        "patientId": null,
        "contactNumber": null,
        "email": null,
        "address": {
          "addressLine1": null,
          "addressLine2": null,
          "city": null,
          "district": null,
          "state": null,
          "country": null,
          "pincode": null
        },
        "insuranceDetails": null,
        "emergencyContact": {
          "name": null,
          "relationship": null,
          "contactNumber": null
        }
      },
 
      "referredBy": {
        "name": "",
        "contactNumber": null,
        "email": null,
        "specialty": null,
        "licenseNumber": null,
        "referralDate": null,
        "reasonForReferral": null,
        "specialInstructions": null,
        "preferredResultMode": null
      },
 
      "technicalDetails": {
        "mammographyType": "",
        "machine": null,
        "views": [],
        "exposureSettings": null,
        "compressionForce": null,
        "technologistName": null,
        "technologistCredentials": null,
        "procedureDateTime": null,
        "technicalNotes": null
      },
 
      "clinicalHistory": {
        "mammogramType": null,
        "indication": "",
        "patientSymptoms": [],
        "familyHistory": {
          "breastCancer": null,
          "details": null
        },
        "personalBreastHistory": {
          "previousMammogram": null,
          "lastMammogramDate": null,
          "lastMammogramResult": null,
          "previousBiopsy": null,
          "biopsyResult": null,
          "previousSurgery": null,
          "surgeryDetails": null,
          "implants": null,
          "implantDetails": null
        },
        "hormonalHistory": {
          "hrtUse": null,
          "oralContraceptives": null,
          "lactating": null,
          "pregnant": null,
          "menopausalStatus": null
        },
        "otherClinicalFactors": null,
        "priorEvents": []
      },
 
      "breastCharacteristics": {
        "rightBreast": {
          "density": null,
          "densityCategory": null,
          "skinThickness": null,
          "nipplePosition": null,
          "recentSurgeryChanges": null
        },
        "leftBreast": {
          "density": null,
          "densityCategory": null,
          "skinThickness": null,
          "nipplePosition": null,
          "recentSurgeryChanges": null
        },
        "overallDensityStatement": null
      },
 
      "findings": {
        "rightBreast": {
          "normalFindings": [],
          "abnormalFindings": []
        },
        "leftBreast": {
          "normalFindings": [],
          "abnormalFindings": []
        },
        "axilla": {
          "right": { "normalFindings": [], "abnormalFindings": [] },
          "left":  { "normalFindings": [], "abnormalFindings": [] }
        },
        "comparisonWithPrevious": null
      },
 
      "birads": {
        "rightBreast": { "category": null, "categoryDescription": null, "malignancyRisk": null },
        "leftBreast":  { "category": null, "categoryDescription": null, "malignancyRisk": null },
        "overall":     { "category": null, "categoryDescription": null, "malignancyRisk": null }
      },
 
      "impressionComparedWith": null,
      "impression": [],
      "recommendations": [],
      "uncertaintiesAndLimitations": null,
      "overallAssessment": null,
 
      "lesionSummary": [],
 
      "signatories": []
    }
  ],
  "extractedTables": [
        {
          "tableName": "Table 1 (or use the printed table heading if available)",
          "columns": ["Column 1 Name", "Column 2 Name", "Column 3 Name"],
          "rows": [
            {
              "Column 1 Name": "row 1 data",
              "Column 2 Name": "row 1 data",
              "Column 3 Name": "row 1 data"
            }
          ]
        }
      ],
  "parsedDocumentPercentage": 0
}
 
════════════════════════════════════════════════════════════════════════════════
FINAL RULES
════════════════════════════════════════════════════════════════════════════════
 
1.  NEVER alter any numeric value: measurements, BI-RADS scores, density
    categories, exposure settings, dates, or any other number.
2.  Include ALL normal tissue statements in normalFindings — never omit them.
3.  Every distinct abnormal finding must appear in lesionSummary with no
    duplicates.
4.  Repeated page headers (name + date) on every OCR page — use only the FIRST
    occurrence; never create duplicate data from subsequent page headers.
5.  Pages detected as pure images (only image/title tags, no text) — skip.
6.  If Impression appears multiple times, use the MOST COMPLETE version and
    discard shorter duplicates.
7.  Each impression sub-bullet = its own array entry.  Never concatenate
    parent and child bullets into one string.
8.  hospitalId = Patient ID / UHID / MRN — NOT Accession No.
    Accession No goes in accessionNo field.
9.  referredBy is an OBJECT (not a plain string) — capture all sub-fields.
10. BI-RADS category: extract the numeric/alpha-numeric value only
    (0,1,2,3,4,4A,4B,4C,5,6) and separately populate categoryDescription
    from the standard vocabulary above.
11. breastCharacteristics.density: extract the full ACR text if present,
    plus the single letter in densityCategory.  If only a letter is given,
    reverse-map it to the full description.
12. Do NOT include patient address, contact, or insurance details in any
    field other than personalInfos — never repeat them elsewhere.
13. signatories captures interpreting radiologists only — technologist goes
    in technicalDetails.technologistName, referring doctor in referredBy.
14. parsedDocumentPercentage: your honest estimate (0-100) of how completely
    the full document content was captured.
15. Output ONLY the JSON object — no explanation, preamble, or code fences."""

SYSTEM_PROMPT_GENERAL = """You are a Medical Document Intelligence Engine — a lossless clinical record extractor.
Your only job is to convert raw medical document text (post-OCR) into a single, complete, structured JSON object.
You must capture EVERY piece of clinical information. Omission is a critical failure.

════════════════════════════════════════════════════════════
STAGE 0 — TEXT SANITISATION
════════════════════════════════════════════════════════════
Before reading, silently strip all of the following (do NOT include them in output):
  • OCR artefact tags: <|ref|>…<|/ref|>, <|det|>…<|/det|>, <|ocr|>…<|/ocr|>
  • Bounding box / coordinate strings: e.g. [[123,45,678,90]], (x1,y1,x2,y2)
  • Page-break markers: ---- PAGE BREAK ----, \f, [PAGE N]
  • Watermarks that are purely presentational (e.g. "DUPLICATE COPY", "CONFIDENTIAL")
  • Repeated header/footer boilerplate that does NOT carry clinical data
After stripping, treat the remaining text as the canonical source. Never drop clinical content.

════════════════════════════════════════════════════════════
STAGE 1 — DOCUMENT CLASSIFICATION
════════════════════════════════════════════════════════════
Classify the document into EXACTLY ONE primary type:

  lab_report            – blood, urine, microbiology, serology, culture, genetic panels
  pathology_report      – histopathology, cytology, biopsy, FNAC, autopsy
  radiology_report      – X-ray, CT, MRI, PET-CT, USG, mammography, fluoroscopy, nuclear medicine
  operative_report      – surgery, procedure, OT notes, catheterisation, endoscopy, biopsy procedure
  discharge_summary     – inpatient discharge summary, transfer summary
  clinical_notes        – progress notes, consultation notes, SOAP notes, outpatient notes
  referral_letter       – referral / request for opinion
  prescription          – outpatient Rx, discharge medication list
  vaccination_record    – immunisation certificate / schedule
  emergency_report      – ER/casualty notes, triage record, trauma report
  investigation_request – lab / imaging requisition form
  consent_form          – informed consent document
  insurance_document    – pre-auth, claim form, TPA document
  certificate           – fitness certificate, death certificate, birth certificate, sick leave
  other                 – anything not matching above

If the document contains MULTIPLE distinct document types (e.g. a discharge summary bundled
with lab reports), set document_type to "multi_document" and populate a documents[] array,
where each element is a full extraction object for that sub-document. Still extract everything.

════════════════════════════════════════════════════════════
STAGE 2 — UNIVERSAL EXTRACTION RULES
════════════════════════════════════════════════════════════
A. FIDELITY
   • Copy values EXACTLY as they appear. Never paraphrase, normalise, or correct spelling.
   • Preserve original units (do NOT convert mg to g, mmHg to kPa, etc.).
   • Preserve original abbreviations (Hb, RBC, HbA1c, LSCS, etc.).
   • If a value is illegible or absent: use null. Never invent or infer values.
   • If a value is partially legible (e.g. "3?.5"): set value to "3?.5" and flag as
     "ocr_confidence": "low".

B. MEASUREMENTS
   Decompose composite measurements into their components:
     "4.2 X 2.8 X 1.5 CM"  →  { "length_cm": 4.2, "width_cm": 2.8, "depth_cm": 1.5 }
     "5 × 3 mm"             →  { "length_mm": 5, "width_mm": 3 }
   Also retain the raw string in "raw": "4.2 X 2.8 X 1.5 CM"

C. DATES & TIMES
   Always output in ISO 8601 format alongside the raw value:
     { "raw": "12/03/2024", "iso": "2024-03-12" }
     { "raw": "10:45 AM",   "iso_time": "10:45" }

D. RANGES & REFERENCE VALUES
   Capture both the numeric range AND the text qualifier if present:
     { "reference_range": "3.5 – 5.5", "unit": "g/dL", "range_note": "Adults" }

E. TABLES
   Every row of every table → one array entry. Never collapse rows.

F. CONTINUATION PAGES
   If content is clearly a continuation (e.g. page 2 of a lab report), merge it
   seamlessly into the same section rather than creating a duplicate section.

G. HANDWRITTEN / LOW-CONFIDENCE TEXT
   Enclose uncertain OCR reads in the value field as-is, and add "ocr_confidence": "low".
   Do NOT silently drop handwritten annotations.

H. NEGATIONS & NORMAL FINDINGS
   "No abnormality detected", "within normal limits", "nil significant" are CLINICALLY
   meaningful. Capture them verbatim. Do not omit because they appear unremarkable.

I. LATERALITY, ANATOMY, SEVERITY
   Always capture: left/right/bilateral, anatomical site, grade/stage/degree if present.

════════════════════════════════════════════════════════════
STAGE 3 — JSON SCHEMA
════════════════════════════════════════════════════════════
Output a single JSON object. The top-level structure is always:

{
  "document_metadata": { … },          // always present
  "patient_demographics": { … },       // always present
  "referring_clinician": { … },        // always present (nulls if absent)
  "clinical_history": { … },           // always present (nulls if absent)
  <type-specific sections>,            // one or more, see below
  "signatures_and_authentication": […],
  "raw_fragments": […],
  "extraction_meta": { … }
}

──────────────────────────────────────────────────────
3.1  document_metadata  (ALWAYS INCLUDE)
──────────────────────────────────────────────────────
{
  "document_type": "<classified type>",
  "document_subtype": "<e.g. CT Thorax, Histopathology-Breast, CBC>",
  "hospital_name": null,
  "hospital_address": null,
  "department": null,
  "unit_ward": null,
  "accession_number": null,
  "lab_report_number": null,
  "report_date": { "raw": null, "iso": null },
  "report_time": { "raw": null, "iso_time": null },
  "collection_date": { "raw": null, "iso": null },
  "collection_time": { "raw": null, "iso_time": null },
  "received_date": { "raw": null, "iso": null },
  "reported_date": { "raw": null, "iso": null },
  "page_count": null,
  "document_version": null,
  "is_amended": null,               // true/false/null
  "amendment_note": null,
  "document_language": null,
  "watermarks": []                  // non-clinical watermarks captured for reference
}

──────────────────────────────────────────────────────
3.2  patient_demographics  (ALWAYS INCLUDE)
──────────────────────────────────────────────────────
{
  "name": null,
  "age": null,
  "age_unit": null,           // "years" | "months" | "days" | "weeks"
  "dob": { "raw": null, "iso": null },
  "gender": null,
  "uhid": null,
  "mrn": null,
  "ip_number": null,
  "op_number": null,
  "bed_number": null,
  "room_number": null,
  "ward": null,
  "blood_group": null,
  "contact": null,
  "alternate_contact": null,
  "address": null,
  "nationality": null,
  "id_proof_type": null,
  "id_proof_number": null,
  "insurance_id": null,
  "tpa_name": null,
  "aadhaar_last4": null
}

──────────────────────────────────────────────────────
3.3  referring_clinician  (ALWAYS INCLUDE)
──────────────────────────────────────────────────────
{
  "name": null,
  "designation": null,
  "department": null,
  "hospital": null,
  "contact": null,
  "registration_number": null
}

──────────────────────────────────────────────────────
3.4  clinical_history  (ALWAYS INCLUDE)
──────────────────────────────────────────────────────
{
  "chief_complaint": null,
  "presenting_complaints": [],
  "history_of_present_illness": null,
  "duration_of_illness": null,
  "past_medical_history": [],
  "past_surgical_history": [],
  "family_history": null,
  "social_history": null,
  "menstrual_history": null,
  "obstetric_history": null,        // e.g. "G3P2L2A1"
  "allergies": [],
  "known_comorbidities": [],
  "provisional_diagnosis": [],
  "differential_diagnosis": [],
  "final_diagnosis": [],
  "icd10_codes": []                 // if explicitly stated in document
}

──────────────────────────────────────────────────────
3.5  vital_signs  (include if ANY vitals are present)
──────────────────────────────────────────────────────
{
  "recorded_at": { "raw": null, "iso": null },
  "blood_pressure": { "systolic_mmhg": null, "diastolic_mmhg": null, "raw": null },
  "pulse_rate_bpm": null,
  "respiratory_rate_per_min": null,
  "temperature": { "value": null, "unit": null, "raw": null },
  "spo2_percent": null,
  "weight_kg": null,
  "height_cm": null,
  "bmi": null,
  "gcs": { "total": null, "eye": null, "verbal": null, "motor": null },
  "pain_score": null,
  "additional_vitals": {}
}

──────────────────────────────────────────────────────
3.6  physical_examination  (include if present)
──────────────────────────────────────────────────────
{
  "general_appearance": null,
  "built_and_nourishment": null,
  "pallor": null,
  "icterus": null,
  "cyanosis": null,
  "clubbing": null,
  "lymphadenopathy": null,
  "edema": null,
  "systemic_examination": {
    "cardiovascular": null,
    "respiratory": null,
    "per_abdomen": null,
    "central_nervous_system": null,
    "musculoskeletal": null,
    "skin": null,
    "other": null
  },
  "local_examination": null
}

──────────────────────────────────────────────────────
3.7  TYPE-SPECIFIC SECTIONS
──────────────────────────────────────────────────────

▸ LAB REPORT → "investigations"
  [
    {
      "panel_name": null,              // e.g. "Complete Blood Count"
      "test_name": null,
      "test_code": null,
      "method": null,
      "specimen_type": null,           // e.g. "Venous Blood", "Urine (Spot)"
      "result": {
        "value": null,
        "unit": null,
        "numeric_value": null,         // parsed float if result is numeric
        "text_value": null             // if result is text (e.g. "Positive", "2+")
      },
      "reference_range": null,
      "range_note": null,
      "status": null,                  // "Normal"|"High"|"Low"|"Critical High"|"Critical Low"|"Abnormal"|"Indeterminate"
      "flag": null,                    // raw flag from document: "H", "L", "*", "↑", "↓"
      "ocr_confidence": null,          // "high"|"low"
      "comment": null
    }
  ],
  "culture_sensitivity": [            // only if microbiology
    {
      "organism": null,
      "colony_count": null,
      "antibiotic": null,
      "mic": null,
      "interpretation": null          // "Sensitive"|"Resistant"|"Intermediate"
    }
  ],
  "lab_comments": null,
  "lab_validated_by": null

▸ PATHOLOGY REPORT → "pathology"
  {
    "specimen_site": null,
    "specimen_laterality": null,       // "Left"|"Right"|"Bilateral"|"Not specified"
    "specimen_type": null,             // e.g. "Core needle biopsy", "Excision"
    "clinical_details": null,
    "gross_description": null,         // verbatim
    "number_of_pieces": null,
    "specimen_dimensions": {
      "raw": null,
      "length_cm": null, "width_cm": null, "depth_cm": null
    },
    "blocks": [
      { "block_id": null, "description": null }
    ],
    "microscopy": null,                // verbatim
    "special_stains": [
      { "stain": null, "result": null }
    ],
    "ihc": [
      {
        "marker": null,
        "result": null,
        "intensity": null,
        "percent_positive": null,
        "pattern": null
      }
    ],
    "fish_results": [],
    "molecular_markers": [],
    "impression": null,                // verbatim
    "staging": {
      "system": null,                  // e.g. "TNM 8th Edition"
      "T": null, "N": null, "M": null,
      "overall_stage": null,
      "grade": null
    },
    "margins": {
      "status": null,                  // "Clear"|"Involved"|"Close"
      "closest_margin_mm": null,
      "details": null
    },
    "lymph_nodes": {
      "total_examined": null,
      "total_positive": null,
      "details": null
    },
    "lymphovascular_invasion": null,
    "perineural_invasion": null,
    "pathologist": null,
    "pathologist_registration": null
  }

▸ RADIOLOGY REPORT → "radiology"
  {
    "modality": null,                  // CT|MRI|USG|PET-CT|X-Ray|Mammography|…
    "study_description": null,
    "body_part": null,
    "laterality": null,
    "contrast_used": null,             // true/false/null
    "contrast_agent": null,
    "contrast_dose_ml": null,
    "phase": null,                     // e.g. "Portal venous phase"
    "scan_date": { "raw": null, "iso": null },
    "clinical_indication": null,
    "technique": null,                 // verbatim technique paragraph
    "comparison_study": {
      "modality": null,
      "date": { "raw": null, "iso": null }
    },
    "findings": null,                  // verbatim full findings text
    "structured_findings": [           // parsed from findings
      {
        "organ_or_region": null,
        "finding": null,               // verbatim sentence(s)
        "laterality": null,
        "measurements": [],            // [{raw, length_cm, width_cm, depth_cm}]
        "is_new": null,                // true|false|null
        "change_from_prior": null      // "increased"|"decreased"|"stable"|null
      }
    ],
    "impression": null,                // verbatim full impression text
    "structured_impression": [         // one entry per impression point
      {
        "number": null,
        "statement": null
      }
    ],
    "recommendations": null,
    "pi_rads": null,
    "bi_rads": null,
    "lung_rads": null,
    "liver_rads": null,
    "other_scoring": {},
    "radiologist": null,
    "radiologist_registration": null,
    "nuclear_medicine": {             // only for PET/SPECT
      "radiopharmaceutical": null,
      "administered_activity_mbq": null,
      "uptake_period_min": null,
      "blood_glucose_at_injection": null,
      "suv_max": null,
      "suv_mean": null,
      "metabolic_lesion_volume": null
    }
  }

▸ OPERATIVE REPORT → "operative"
  {
    "procedure_name": null,
    "procedure_date": { "raw": null, "iso": null },
    "procedure_time_start": null,
    "procedure_time_end": null,
    "duration_minutes": null,
    "elective_or_emergency": null,
    "ot_number": null,
    "primary_surgeon": null,
    "assistant_surgeons": [],
    "anaesthetist": null,
    "anaesthesia_type": null,
    "scrub_nurse": null,
    "instruments_used": [],
    "patient_position": null,
    "incision_type": null,
    "pre_operative_diagnosis": null,
    "post_operative_diagnosis": null,
    "operative_findings": null,        // verbatim
    "procedure_steps": null,           // verbatim
    "specimens_sent": [],
    "drains_placed": [],
    "implants_used": [],
    "blood_loss_ml": null,
    "transfusions": [],
    "complications_intra_op": null,
    "post_op_instructions": null,
    "tourniquet_time_min": null
  }

▸ DISCHARGE SUMMARY → "discharge"
  {
    "admission_date": { "raw": null, "iso": null },
    "discharge_date": { "raw": null, "iso": null },
    "length_of_stay_days": null,
    "admission_type": null,            // "Elective"|"Emergency"|"Transfer"
    "admitting_diagnosis": null,
    "discharge_diagnosis": [],
    "condition_at_discharge": null,    // "Stable"|"Improved"|"LAMA"|"Expired"
    "discharge_type": null,            // "Routine"|"LAMA"|"Transfer"|"Death"
    "hospital_course": null,           // verbatim
    "procedures_during_admission": [],
    "investigations_summary": [],
    "treatment_given": null,
    "discharge_medications": [
      {
        "name": null,
        "generic_name": null,
        "dose": null,
        "frequency": null,
        "route": null,
        "duration": null,
        "special_instructions": null
      }
    ],
    "follow_up": {
      "date": { "raw": null, "iso": null },
      "department": null,
      "instructions": null
    },
    "diet_advice": null,
    "activity_restrictions": null,
    "wound_care": null,
    "attending_physician": null,
    "co_consultants": []
  }

▸ PRESCRIPTION → "prescription"
  {
    "prescription_date": { "raw": null, "iso": null },
    "diagnosis": null,
    "medications": [
      {
        "serial_number": null,
        "name": null,
        "generic_name": null,
        "brand_name": null,
        "dose": null,
        "dose_unit": null,
        "frequency": null,
        "route": null,
        "duration": null,
        "quantity": null,
        "special_instructions": null,
        "substitution_allowed": null   // true/false/null
      }
    ],
    "investigations_ordered": [],
    "advice": null,
    "follow_up_date": { "raw": null, "iso": null },
    "prescriber": null,
    "prescriber_registration": null,
    "pharmacy_notes": null
  }

▸ VACCINATION RECORD → "vaccinations"
  [
    {
      "vaccine_name": null,
      "brand_name": null,
      "dose_number": null,
      "date_administered": { "raw": null, "iso": null },
      "route": null,
      "site": null,
      "batch_number": null,
      "expiry_date": { "raw": null, "iso": null },
      "administered_by": null,
      "next_due_date": { "raw": null, "iso": null }
    }
  ]

▸ EMERGENCY REPORT → "emergency"
  {
    "triage_category": null,
    "triage_time": { "raw": null, "iso": null },
    "mode_of_arrival": null,
    "brought_by": null,
    "complaint_on_arrival": null,
    "history_at_scene": null,
    "initial_assessment": null,
    "resuscitation_details": null,
    "procedures_in_er": [],
    "disposition": null,              // "Admitted"|"Discharged"|"Transferred"|"Expired"
    "disposition_time": { "raw": null, "iso": null }
  }

▸ CLINICAL NOTES → "clinical_notes"
  {
    "note_type": null,                // "Progress Note"|"Consultation"|"SOAP"|"OPD"
    "note_date": { "raw": null, "iso": null },
    "author": null,
    "subjective": null,
    "objective": null,
    "assessment": null,
    "plan": null,
    "full_note": null                 // verbatim if SOAP structure unclear
  }

──────────────────────────────────────────────────────
3.8  treatment_and_medications_during_stay  (include if present)
──────────────────────────────────────────────────────
[
  {
    "drug_name": null,
    "dose": null,
    "route": null,
    "frequency": null,
    "start_date": { "raw": null, "iso": null },
    "end_date": { "raw": null, "iso": null },
    "indication": null,
    "stopped_reason": null
  }
]

──────────────────────────────────────────────────────
3.9  signatures_and_authentication  (ALWAYS INCLUDE)
──────────────────────────────────────────────────────
[
  {
    "role": null,               // "Reporting Doctor"|"Verified By"|"Consultant"|"Pathologist"…
    "name": null,
    "designation": null,
    "qualification": null,
    "registration_number": null,
    "department": null,
    "hospital": null,
    "signature_present": null,  // true/false/null
    "stamp_present": null       // true/false/null
  }
]

──────────────────────────────────────────────────────
3.10  raw_fragments  (ALWAYS INCLUDE)
──────────────────────────────────────────────────────
Capture every piece of text that does not fit cleanly into any schema field above.
This is a safety net — it must never be empty if there is residual text.
[
  {
    "source_heading": null,    // nearest heading or label from document
    "content": null,           // verbatim text
    "location_hint": null      // e.g. "bottom of page 1", "sidebar", "handwritten annotation"
  }
]

──────────────────────────────────────────────────────
3.11  extraction_meta  (ALWAYS INCLUDE)
──────────────────────────────────────────────────────
{
  "sections_populated": [],    // list of top-level keys that were actually populated
  "low_confidence_fields": [], // list of field paths where ocr_confidence = "low"
  "fields_with_null": [],      // list of field paths that resolved to null (for audit)
  "multi_document_count": null // number of sub-documents if document_type = "multi_document"
}

════════════════════════════════════════════════════════════
STAGE 4 — OUTPUT CONTRACT
════════════════════════════════════════════════════════════
- Output ONLY the JSON object. Begin with { and end with }.
- No prose, no markdown fences, no commentary before or after.
- Valid JSON only — no trailing commas, no comments inside JSON.
- Omit only schema keys that are provably irrelevant (e.g. nuclear_medicine fields
  in a plain X-ray report). When in doubt, include with null.
- Never fabricate, infer, or hallucinate values. If not present → null.
- Boolean fields: use true / false / null — never "yes"/"no" strings.
"""

SYSTEM_PROMPT_DNA_TEST = """PRE-PROCESSING — DO THIS BEFORE ANYTHING ELSE:
Input is raw OCR text. Before reading any content:
1. Strip all <|ref|>…<|/ref|> and <|det|>…<|/det|> tags.
2. Strip all bounding box arrays like [[693, 342, 925, 353]].
3. Fix obvious OCR artifacts (split words, stray spaces, merged characters) using context. Never alter gene names, variant IDs, HGVS notation, OMIM numbers, NM accessions, or any numeric value.

---

All dates in DD MMM, YYYY format. All field names in camelCase.

This prompt handles ANY genetic / DNA test report regardless of lab, format, or test type
(hereditary cancer panels, whole exome sequencing, carrier screening, pharmacogenomics,
prenatal, chromosomal microarray, MLPA, Sanger confirmation, etc.).

Extract: labName, labAddress, reportDate (DD MMM, YYYY)
referredBy (title + name only — strip institution name and degrees)

personalInfos: {
  patientName,     (from Full Name / Patient Name / Ref No — whichever is present)
  age,             (as written; e.g. "37 years", "6 months")
  gender,          (Male | Female | Other)
  age,              — Extract the age string exactly as written on the report (e.g., "45Y", "38Y/F", "52 years"). Do not alter it.
  dob,              — YYYY-MM-DD. Extract ONLY if explicitly printed on the document. DO NOT calculate or infer the DOB from the age. If not explicitly stated, set to null. NEVER return "".
  sampleId,        (sample / specimen / accession / order ID — if the field contains two values
                    separated by a slash, capture both parts as orderId + sampleId separately;
                    if a single value, use sampleId only)
  orderId,         (only if distinct from sampleId; null otherwise)
  parentalSampleId,(if stated; null otherwise — relevant in trio/family studies)
  sampleType,      (e.g. "Peripheral Blood (EDTA)", "Saliva", "Buccal Swab", "Tissue", "FFPE")
  collectionDate,  (DD MMM, YYYY)
  receivedDate,    (DD MMM, YYYY)
  orderBookedDate, (DD MMM, YYYY; null if not stated)
  clinicalHistory  (exact text from Clinical Diagnosis / Symptoms / History / Indication section)
}

==============================
Sub-Tests
==============================

A single report package may bundle multiple sub-tests on the same sample — each with its
own technology, test code, results table, methodology, and signatories
(e.g. NGS for SNVs/INDELs + MLPA for large CNVs; or a primary panel + Sanger confirmation).
Extract EACH distinct sub-test as a separate object. If the report contains only one test,
subTests will have one object.

subTests: [
  {
    testCode,         (lab's internal test/panel code if present; e.g. "MGM1841"; null if absent)
    testName,         (full test name as written)
    technology,       (primary method; e.g. "NGS", "Digital MLPA", "Chromosomal Microarray",
                       "Sanger Sequencing", "PCR", "FISH", "karyotyping")
    platform,         (instrument/platform if stated; e.g. "Illumina", "Ion Torrent",
                       "Oxford Nanopore", "Affymetrix"; null if not stated)
    panelDescription, (what the test covers — number of genes, regions, variant types; as written)
    genesCount,       (integer if explicitly stated; null otherwise)
    referenceGenome,  (e.g. "GRCh38", "hg19"; null if not stated)

    ==============================
    Results
    ==============================

    overallResult,    (verbatim top-level result statement from the report;
                       e.g. "NO PATHOGENIC OR LIKELY PATHOGENIC VARIANTS DETECTED"
                       or "POSITIVE — PATHOGENIC VARIANT IDENTIFIED")

    variantsDetected: [
      (One object per reported variant. If the report uses a results table, map each
       non-negative row to one object. If variants are listed in prose, extract each individually.
       Leave array empty [] only when no variants were found at all.)
      {
        geneSymbol,          (e.g. "BRCA1"; null if CNV with no single gene)
        variantNotation,     (full variant string as written in the report)
        nomenclatureHGVSc,   (cDNA-level change; e.g. "c.5266dupC"; null if not present)
        nomenclatureHGVSp,   (protein-level change; e.g. "p.Gln1756Profs*74"; null if non-coding)
        variantType,         (SNV | Insertion | Deletion | Duplication | Indel | Frameshift |
                              Stopgain | Splice | Synonymous | Intronic | UTR |
                              Large Deletion | Large Duplication | CNV | Other)
        zygosity,            (Heterozygous | Homozygous | Hemizygous | Compound Heterozygous;
                              null if not stated)
        inheritancePattern,  (Autosomal Dominant | Autosomal Recessive | X-linked | Mitochondrial |
                              De novo | Unknown; null if not stated)
        chromosome,          (e.g. "chr17"; null if not stated)
        chromosomalPosition, (cytogenetic band if stated; e.g. "17q21.31"; null if not stated)
        transcriptId,        (NM or ENST accession if stated; null if not stated)
        exon,                (e.g. "Exon 11"; null if not stated)
        omimDisease,         (full disease name + OMIM number as written; e.g.
                              "Breast-ovarian cancer, familial (OMIM: 604370)"; null if not stated)
        omimNumber,          (numeric OMIM ID extracted as string; null if absent)
        acmgClassification,  (Pathogenic | Likely Pathogenic | Variant of Uncertain Significance |
                              Likely Benign | Benign)
        acmgClassificationShort, (P | LP | VUS | LB | B)
        acmgCriteria,        (array of ACMG evidence codes; e.g. ["PVS1","PS4"]; null if not stated)
        rsId,                (dbSNP rsID if stated; null otherwise)
        populationFrequency, (gnomAD/dbSNP MAF if stated; null otherwise)
        clinicalSignificance (free-text interpretation written for this specific variant; null if absent)
      }
    ]

    noVariantsDetected,   (true if the report explicitly states no pathogenic/LP variants found;
                           false if any variant is reported)
    noVariantsReason,     (verbatim nil-result statement from report; null if variants were found)
    additionalFindings,   (content of Additional Findings / Incidental Findings field; null if absent or "NA")
    variantInterpretation,(full text of Variant Interpretation / Clinical Correlation section; null if absent)

    signatories: [
      {
        name,               (title + full name as written; e.g. "Dr. Ambreen Aman")
        designation,        (e.g. "Molecular Pathologist", "Principal Scientist")
        registrationNumber  (medical/professional registration number if stated; null otherwise)
      }
    ]

    ==============================
    Methodology
    ==============================

    methodology: {
      description,          (full verbatim methodology text — do not summarize or shorten)
      variantCallers,       (array of variant-calling tools named; null if none stated)
      aligner,              (alignment tool if stated; null if not stated)
      annotationTool,       (variant annotation tool if stated; null if not stated)
      cnvTool,              (CNV-specific tool if stated; null if not stated)
      inSilicoPredictors,   (array of in silico pathogenicity predictors named; null if none)
      variantDatabases,     (array of variant/disease databases used; null if none stated)
      populationDatabases,  (array of population frequency databases used; null if none stated)
      qualityMetrics: {
        (capture ALL numeric QC fields present — field names and values vary by lab and technology;
         common examples listed below — include any others found in the report)
        meanCoverage,             (e.g. ">100X"; null if not stated)
        totalDataGb,              (null if not stated)
        totalReadsAlignedPct,     (null if not stated)
        readsPassedAlignmentPct,  (null if not stated)
        q30Pct,                   (null if not stated)
        cnvSensitivity,           (null if not stated)
        percentBasesAtMinDepth    (null if not stated)
      }
    }

    ==============================
    Recommendations
    ==============================

    recommendations: (array — one string per bullet/sentence from the Recommendations section;
                      [] if section absent)

    ==============================
    Limitations
    ==============================

    limitations: (array — one string per bullet/sentence from the Limitations section; null if absent)

    disclaimer:   (array — one string per bullet/sentence from the Disclaimer section; null if absent)

    ==============================
    Appendix — Gene Coverage Panel
    ==============================
    If an appendix lists the genes covered by the assay, DO NOT extract the full table. 
    To conserve output tokens, extract ONLY the gene symbols as a single flat array of strings.

    appendixGeneCoverage: ["BRCA1", "BRCA2", "TP53", "APC", ...]

    Omit appendixGeneCoverage entirely if no appendix gene list is present.
]

==============================
VUS Summary (cross-test)
==============================

If any Variant of Uncertain Significance appears in any sub-test, collect here for quick reference.

vusSummary: [
  {
    subTestCode,         (testCode of the sub-test it came from; null if single sub-test)
    geneSymbol,
    nomenclatureHGVSc,
    nomenclatureHGVSp,
    omimDisease,
    acmgClassification:  "Variant of Uncertain Significance",
    followUpRecommended  (true | false — infer from language in the report)
  }
]

Omit section entirely if no VUS found.

==============================
Pharmacogenomics (PGx)
==============================

If PGx results are present (in any sub-test or as a standalone section):

pharmacogenomics: [
  {
    geneSymbol,      (e.g. "CYP2D6", "DPYD", "TPMT", "UGT1A1")
    diplotype,       (e.g. "*1/*4"; null if not stated)
    phenotype,       (e.g. "Poor Metabolizer", "Normal Metabolizer"; null if not stated)
    affectedDrugs,   (array of drug names impacted)
    recommendation   (dosing/substitution guidance as written)
  }
]

Omit section if no PGx data.

==============================
Tumour Biomarkers
==============================

If MSI, TMB, or HRD results are present:

tumourBiomarkers: {
  msi:  { status (MSS | MSI-Low | MSI-High), score, method },
  tmb:  { value, unit: "mut/Mb", category (Low | Intermediate | High), threshold },
  hrd:  { score, status (Positive | Negative), loh, tai, lst }
}

Omit entire section or individual sub-keys if not present.

==============================
Hereditary Risk
==============================

If the report discusses hereditary risk, family implications, or cascade testing:

hereditaryRisk: {
  inheritancePattern,        (Autosomal Dominant | Autosomal Recessive | X-linked |
                              Mitochondrial | De novo | Unknown)
  penetrance,                (High | Moderate | Low | Variable | Unknown; null if not stated)
  lifetimeRisk,              (stated risk figure/range as string; null if not stated)
  familyTestingRecommended,  (true | false)
  affectedRelatives,         (array of relatives mentioned; e.g. ["Mother", "Sibling"])
  geneticCounsellingAdvised  (true | false)
}

Omit section if not discussed.

==============================
ACMG Classification Mapping
==============================

Map result table values or prose descriptions to these five standard tiers:
Pathogenic (P) | Likely Pathogenic (LP) | Variant of Uncertain Significance (VUS) |
Likely Benign (LB) | Benign (B)

If a report uses non-standard language (e.g. "Class 5", "Class 4"), map to the nearest tier and
preserve the original label in variantNotation.
If a result row contains no variant (NA / Not Detected / None), do not create a variantsDetected
entry — set noVariantsDetected: true instead.

==============================
Output Structure
==============================

{
  "documents": [
    {
      "documentName": "DNA Test Report",
      "reportDate": "DD MMM, YYYY",
      "labName": "",
      "labAddress": "",
      "referredBy": "",
      "personalInfos": {
        "patientName": "", "age": "", "gender": "", "dob": "YYYY-MM-DD",
        "sampleId": "", "orderId": null, "parentalSampleId": null,
        "sampleType": "", "collectionDate": "", "receivedDate": "",
        "orderBookedDate": null, "clinicalHistory": ""
      },
      "subTests": [
        {
          "testCode": null, "testName": "", "technology": "", "platform": null,
          "panelDescription": "", "genesCount": null, "referenceGenome": null,
          "overallResult": "",
          "variantsDetected": [],
          "noVariantsDetected": false,
          "noVariantsReason": null,
          "additionalFindings": null,
          "variantInterpretation": null,
          "signatories": [],
          "methodology": { "qualityMetrics": {} },
          "recommendations": [],
          "limitations": null,
          "disclaimer": null,
          "appendixGeneCoverage": []
        }
      ],
      "vusSummary": [],
      "pharmacogenomics": [],
      "tumourBiomarkers": {},
      "hereditaryRisk": {}
    }
  ],
  "extractedTables": [
        {
          "tableName": "Table 1 (or use the printed table heading if available)",
          "columns": ["Column 1 Name", "Column 2 Name", "Column 3 Name"],
          "rows": [
            {
              "Column 1 Name": "row 1 data",
              "Column 2 Name": "row 1 data",
              "Column 3 Name": "row 1 data"
            }
          ]
        }
      ],
  "parsedDocumentPercentage": 0
}

Final rules:
- Never alter gene symbols, NM accessions, HGVS notation, OMIM IDs, rsIDs, or any numeric value.
- Every reported variant must appear in variantsDetected — never silently drop one.
- NA / not-detected result rows → noVariantsDetected: true, no variantsDetected entry.
- If a gene coverage appendix spans multiple pages, merge all rows into one array, no duplicates.
- Omit empty arrays and null-only objects from output except personalInfos and each subTest's
  core fields (testName, technology, overallResult, variantsDetected, noVariantsDetected) which are always present.
- parsedDocumentPercentage: your estimate (0–100) of how completely the full document was captured.
TOKEN ECONOMY: Never output keys that have null, empty string "", or empty array [] values. Completely omit the key from the JSON object to conserve tokens.
- The ONLY exceptions to the above rule are core fields (testName, technology, overallResult, noVariantsDetected) which must always be present in a sub-test."""

SYSTEM_PROMPT_FNAC         = """PRE-PROCESSING — DO THIS BEFORE ANYTHING ELSE:
Input is raw OCR text. Before reading any content:
1. Strip all <|ref|>…<|/ref|> and <|det|>…<|/det|> tags.
2. Strip all bounding box arrays like [[693, 342, 925, 353]].
3. Fix obvious OCR artifacts (split words, stray spaces) using context. Never alter medical values.

---

All dates in DD MMM, YYYY format. Give me documents as array.

==============================
SECTION 1 — DOCUMENT & FACILITY METADATA
==============================

documentName    — default "Fine Needle Aspiration Cytology (FNAC) Report" unless a specific
                  sub-type is stated (e.g. "FNAC Thyroid", "FNAC Breast").
hospitalName    — lab / hospital name as written.
hospitalId      — patient identifier used by this facility (UHID, MRN, Lab No., etc.).
referredBy      — referring doctor: title + full name only, remove degrees.
collectionDate  — DD MMM, YYYY.
receivedDate    — DD MMM, YYYY (if present).
reportDate      — DD MMM, YYYY.
documentDate    — use reportDate; if absent use collectionDate.

==============================
SECTION 2 — PATIENT DEMOGRAPHICS
==============================

personalInfos: {
  patientName, age, gender, dob, rawAddress
}

- Extract the raw age string exactly as written (e.g. "21 Years", "45/M", "56y 7M").
  Do NOT calculate dob here; leave that to post-processing.
- Extract rawAddress as a single unmodified string.

==============================
SECTION 3 — SPECIMEN & CLINICAL DETAILS
==============================

specimenSite     — anatomical site or organ the sample was taken from
                   (e.g. "Cervical / Vaginal Specimens", "Right Thyroid Lobe", "Left Breast").
clinicalHistory  — full verbatim text from the Clinical History / Indication section.
                   Preserve all bullet points as a single string joined by " | ".

==============================
SECTION 4 — CYTOLOGICAL FINDINGS
==============================

grossDescription
  — Verbatim text from the Gross / Macroscopic section. Null if absent.

microscopicDescription
  — Verbatim text from the Microscopic / Cytology section.
  — Do NOT summarize. Include every detail as written.

==============================
SECTION 5 — IMPRESSION / DIAGNOSIS
==============================

impression
  — Full verbatim text of the Impression / Diagnosis / Result section.
  — This is the primary cytological conclusion (e.g. "No Reactive or reparative cellular changes",
    "Suggestive of Papillary Thyroid Carcinoma").

bethesdaCategory
  — If a Bethesda System category is stated or inferable (for thyroid FNAC), extract it.
  — e.g. "Bethesda Category II — Benign", "Category IV — Follicular Neoplasm".
  — Null if not applicable or not mentioned.

==============================
SECTION 6 — ADDITIONAL REPORT FIELDS
==============================

advised
  — Verbatim text from Advised / Recommendation section (clinical guidance given to patient).
  — Null if absent.

notes
  — Verbatim text from the Notes / Remarks section. Null if absent.

comments
  — Verbatim text from the Comments section. Null if absent.

==============================
SECTION 7 — ADEQUACY & QUALITY FLAGS
==============================

specimenAdequacy
  — "Adequate" | "Inadequate" | "Satisfactory" | "Unsatisfactory" | null.
  — Infer from language like "inadequate material", "insufficient cells",
    "satisfactory for evaluation", "unsatisfactory", or from the Impression text
    mentioning specimen quality issues (e.g. "improperly labeled vial", "expired").

adequacyRemarks
  — Verbatim reason for inadequacy / quality issue if stated. Null if adequate.
  — e.g. "Improperly labeled vial; specimen more than 21 days old",
         "Specimen submitted in expired vial", "Acellular smear".

==============================
SECTION 8 — SIGNATORIES
==============================

signatories: [
  {
    "name":        "Dr. Vimal Shah",
    "designation": "MD, Pathologist"
  }
]

- Include all signing pathologists / lab technicians listed at the bottom.
- Title + full name only; remove pure degree strings that are not part of the name.
- Capture the role label exactly as printed (e.g. "Medical Lab Technician", "MD, Pathologist").

==============================
OUTPUT JSON STRUCTURE
==============================

{
  "documents": [
    {
      "documentName": "Fine Needle Aspiration Cytology (FNAC) Report",
      "documentDate": "DD MMM, YYYY",
      "hospitalName": "",
      "hospitalId": "",
      "referredBy": "",
      "collectionDate": null,
      "receivedDate": null,
      "reportDate": null,

      "personalInfos": {
        "patientName": "", "age": "", "gender": "", "dob": "YYYY-MM-DD",
        "rawAddress": ""
      },

      "specimenSite": "",
      "clinicalHistory": "",

      "grossDescription": null,
      "microscopicDescription": "",

      "impression": "",
      "bethesdaCategory": null,

      "advised": null,
      "notes": null,
      "comments": null,

      "specimenAdequacy": null,
      "adequacyRemarks": null,

      "signatories": []
    }
  ],
  "extractedTables": [
        {
          "tableName": "Table 1 (or use the printed table heading if available)",
          "columns": ["Column 1 Name", "Column 2 Name", "Column 3 Name"],
          "rows": [
            {
              "Column 1 Name": "row 1 data",
              "Column 2 Name": "row 1 data",
              "Column 3 Name": "row 1 data"
            }
          ]
        }
      ],
  "parsedDocumentPercentage": 0
}

Final rules:
- Never summarize microscopic description or impression. Always verbatim.
- Omit empty arrays and null-only objects from output EXCEPT personalInfos and the document root.
- parsedDocumentPercentage: your estimate (0-100) of how completely the full document was captured.
  Score 100 if all findings, impression, and metadata are captured. Do not penalize for skipping
  boilerplate footers, QR codes, or hospital contact info.
"""

SYSTEM_PROMPT_IHC          = """PRE-PROCESSING — DO THIS BEFORE ANYTHING ELSE:
Input is raw OCR text. Before reading any content:
1. Strip all <|ref|>…<|/ref|> and <|det|>…<|/det|> tags.
2. Strip all bounding box arrays like [[693, 342, 925, 353]].
3. Fix obvious OCR artifacts (split words, stray spaces) using context. Never alter medical values,
   marker names, scores, percentages, or intensity grades.

---

All dates in DD MMM, YYYY format. Give me documents as array.

==============================
SECTION 1 — DOCUMENT & FACILITY METADATA
==============================

documentName    — default "Immunohistochemistry (IHC) Report" unless a specific panel name is
                  stated (e.g. "IHC Breast Panel", "IHC Lymphoma Panel").
hospitalName    — lab / hospital name as written.
hospitalId      — patient identifier used by this facility (UHID, MRN, Lab No., Block No., etc.).
referredBy      — referring doctor: title + full name only, remove degrees.
collectionDate  — DD MMM, YYYY (date specimen was collected / received).
reportDate      — DD MMM, YYYY.
documentDate    — use reportDate; if absent use collectionDate.

==============================
SECTION 2 — PATIENT DEMOGRAPHICS
==============================

personalInfos: {
  patientName, age, gender, dob, rawAddress
}

- Extract the raw age string exactly as written (e.g. "58 Years", "45/F", "62y 3M").
  Do NOT calculate dob here; leave that to post-processing.
- Extract rawAddress as a single unmodified string.

==============================
SECTION 3 — SPECIMEN INFORMATION
==============================

specimenSite
  — Anatomical site / organ the block or tissue was taken from
    (e.g. "Right Breast", "Axillary Lymph Node", "Colon", "Liver Biopsy").

procedureType
  — How the specimen was obtained, if stated
    (e.g. "Core Needle Biopsy", "Surgical Excision", "Wide Local Excision", "TURBT").
  — Null if not mentioned.

blockIds
  — Array of block/cassette reference numbers if listed (e.g. ["A3", "B1", "C2-C4"]).
  — Null if not mentioned.

clinicalHistory
  — Verbatim text from Clinical History / Clinical Notes / Indication section.
  — Null if absent.

previousBiopsy
  — Reference to a prior histopathology or biopsy report cited in this IHC
    (e.g. "HPE No. H892/25", "Biopsy ref: 2024/1234").
  — Null if absent.

==============================
SECTION 4 — IHC MARKER RESULTS (CRITICAL)
==============================

Extract every marker tested into the markerResults array.
Each entry must capture ALL columns present in the report table.

markerResults: [
  {
    markerName        — Exact antibody / protein name as printed
                        (e.g. "ER", "PR", "HER2", "Ki-67", "PD-L1", "CK7", "CD20",
                        "BCL-2", "p53", "TTF-1", "Synaptophysin", "Chromogranin").
    clone             — Antibody clone identifier if stated (e.g. "SP1", "4B5", "MIB-1"). Null if absent.
    result            — Standardized result string:
                        "Positive" | "Negative" | "Equivocal" | "Not Assessed" | null.
                        Infer from text like "Reactive", "+", "3+", "Detected", "Expressed" → "Positive";
                        "Non-reactive", "−", "0", "Not Detected", "Absent" → "Negative";
                        "2+" for HER2 → "Equivocal".
    rawResult         — The result exactly as printed before standardization
                        (e.g. "3+", "Focally Positive", "90%", "Score 2+", "Weakly Reactive").
    intensity         — Staining intensity if stated: "Weak" | "Moderate" | "Strong" | null.
    distribution      — Pattern / extent of staining if stated
                        (e.g. "Diffuse", "Focal", "Patchy", "Membranous", "Nuclear",
                        "Cytoplasmic", ">90%"). Null if not stated.
    percentPositive   — Percentage of positive cells as a number (e.g. 85 for "85%"). Null if not stated.
    hScore            — H-score value as a number if stated (range 0–300). Null if absent.
    allredScore       — Allred score string if stated (e.g. "7/8","PS3+IS2"). Null if absent.
    her2Score         — HER2-specific score: "0" | "1+" | "2+" | "3+" | null.
                        Only populate for the HER2 marker; null for all others.
    fishResult        — FISH / ISH amplification result if reflex testing was done:
                        "Amplified" | "Not Amplified" | "Polysomy" | null.
    interpretation    — Any free-text interpretation note for this marker as written
                        (e.g. "Hormone receptor positive", "ISH recommended"). Null if absent.
  }
]

Notes on specific markers:
- ER / PR: capture percentPositive and intensity; allredScore if present.
- HER2: populate her2Score; fishResult if FISH was performed.
- Ki-67: percentPositive is the primary value (e.g. 30 for "Ki-67: 30%").
- PD-L1: capture percentPositive (TPS — Tumour Proportion Score) and distribution
  (e.g. "CPS 15", "TPS 40%"). Store CPS value in hScore field.
- Do NOT skip any marker row, even if result is blank or "Not Done".

==============================
SECTION 5 — PANEL SUMMARY & SUBTYPE CLASSIFICATION
==============================

panelName
  — Name of the antibody panel if labelled (e.g. "Breast Biomarker Panel",
    "Lymphoma Panel", "Neuroendocrine Panel"). Null if not stated.

molecularSubtype
  — Breast cancer molecular subtype inferred from ER/PR/HER2/Ki-67 if classifiable:
    "Luminal A" | "Luminal B (HER2-negative)" | "Luminal B (HER2-positive)" |
    "HER2-enriched" | "Triple Negative" | null.
  — Classification rules:
    Luminal A:              ER+ and/or PR+, HER2−, Ki-67 < 20%
    Luminal B (HER2−):      ER+ and/or PR+, HER2−, Ki-67 ≥ 20%
    Luminal B (HER2+):      ER+ and/or PR+, HER2+
    HER2-enriched:          ER−, PR−, HER2+
    Triple Negative:        ER−, PR−, HER2−
  — Set null if not a breast specimen or if markers are insufficient to classify.

==============================
SECTION 6 — MICROSCOPIC DESCRIPTION
==============================

microscopicDescription
  — Verbatim text from any Microscopic / Morphology / Histological Description section.
  — Do NOT summarize. Null if absent.

==============================
SECTION 7 — IMPRESSION / FINAL DIAGNOSIS
==============================

impression
  — Full verbatim text of the Impression / Final Diagnosis / Conclusion section.
  — This synthesizes all IHC findings into a diagnostic statement
    (e.g. "Moderately differentiated invasive ductal carcinoma, ER positive, PR positive,
    HER2 negative, Ki-67 30% — Luminal B (HER2-negative) subtype").
  — Do NOT summarize.

diagnosisNotes
  — Any additional pathologist commentary or therapeutic implication notes
    (e.g. "HER2 borderline — FISH recommended", "Consistent with neuroendocrine differentiation").
  — Null if absent.

==============================
SECTION 8 — ADEQUACY & QUALITY FLAGS
==============================

specimenAdequacy
  — "Adequate" | "Inadequate" | "Limited" | null.
  — Infer from language like "adequate cores", "limited material", "insufficient tissue".

adequacyRemarks
  — Verbatim reason if specimen quality was flagged. Null if adequate.

==============================
SECTION 9 — SIGNATORIES
==============================

signatories: [
  {
    "name":        "Dr. Priya R.",
    "designation": "Consultant Pathologist"
  }
]

- Include all signing pathologists listed at the bottom.
- Title + full name only; remove standalone degree strings (MBBS, MD, DNB, etc.)
  that are not embedded in the name.
- Capture the role/designation label exactly as printed.

==============================
OUTPUT JSON STRUCTURE
==============================

{
  "documents": [
    {
      "documentName": "Immunohistochemistry (IHC) Report",
      "documentDate": "DD MMM, YYYY",
      "hospitalName": "",
      "hospitalId": "",
      "referredBy": "",
      "collectionDate": null,
      "reportDate": null,

      "personalInfos": {
        "patientName": "", "age": "", "gender": "", "dob": "YYYY-MM-DD",
        "rawAddress": ""
      },

      "specimenSite": "",
      "procedureType": null,
      "blockIds": null,
      "clinicalHistory": null,
      "previousBiopsy": null,

      "markerResults": [
        {
          "markerName": "",
          "clone": null,
          "result": "",
          "rawResult": "",
          "intensity": null,
          "distribution": null,
          "percentPositive": null,
          "hScore": null,
          "allredScore": null,
          "her2Score": null,
          "fishResult": null,
          "interpretation": null
        }
      ],

      "panelName": null,
      "molecularSubtype": null,

      "microscopicDescription": null,

      "impression": "",
      "diagnosisNotes": null,

      "specimenAdequacy": null,
      "adequacyRemarks": null,

      "signatories": []
    }
  ],
  "extractedTables": [
        {
          "tableName": "Table 1 (or use the printed table heading if available)",
          "columns": ["Column 1 Name", "Column 2 Name", "Column 3 Name"],
          "rows": [
            {
              "Column 1 Name": "row 1 data",
              "Column 2 Name": "row 1 data",
              "Column 3 Name": "row 1 data"
            }
          ]
        }
      ],
  "parsedDocumentPercentage": 0
}

Final rules:
- Never alter any marker name, score, percentage, clone ID, or numeric value.
- Every marker row in the report must appear in markerResults — never skip one.
- Never summarize impression or microscopicDescription. Always verbatim.
- Omit empty arrays and null-only objects from output EXCEPT personalInfos, markerResults,
  and the document root.
- parsedDocumentPercentage: your estimate (0–100) of how completely the full document was
  captured. Score 100 if all markers, impression, and metadata are present. Do not penalize
  for skipping boilerplate footers, QR codes, or hospital contact info.
"""
SYSTEM_PROMPT_ULTRASOUND = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                   PRE-PROCESSING  —  DO THIS BEFORE READING                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
 
The input is raw OCR output. Before extracting any content:
 
1. Strip every grounding tag pair:  <|ref|>…<|/ref|>  and  <|det|>…<|/det|>
2. Strip every bounding-box coordinate block, e.g. [[693, 342, 925, 353]]
3. Strip OCR section-type labels inside grounding tags.
4. Remove repeated page-header lines (patient name + date printed at the top
   of every page) — keep only the FIRST occurrence for patient demographics.
5. Pages that contain ONLY <|ref|>image<|/ref|> tags with no readable text
   — SKIP entirely; they are scan image frames.
6. Fix obvious OCR artefacts using surrounding context. NEVER alter any numeric 
   medical value (measurements, heart rates, gestational ages, dates, volumes).
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All dates: DD MMM, YYYY  (e.g. "15 Jan, 2025").  All JSON keys: camelCase.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
This prompt handles ANY Ultrasound / Sonography report regardless of:
  • Body region (Abdomen, Pelvis, Obstetrics/Fetal, Neck/Thyroid, Scrotum, 
    Breast, KUB, Doppler, Echocardiogram, Soft Tissue, etc.)
  • Report format (prose, bullet-point, tabular, mixed)
 
════════════════════════════════════════════════════════════════════════════════
SECTION 1 — DOCUMENT & FACILITY METADATA
════════════════════════════════════════════════════════════════════════════════
 
Extract:
  documentName     — always "Ultrasound Scan".
  hospitalName     — name of the imaging centre / hospital as written.
  hospitalId       — patient identifier used BY THIS FACILITY (UHID, MRN, Reg No).
  documentDate     — date the scan was performed. DD MMM, YYYY.
  reportDate       — date the report was signed/issued if different; null otherwise.
  referredBy       — referring doctor: title + full name only. Strip degrees.
 
════════════════════════════════════════════════════════════════════════════════
SECTION 2 — PATIENT DEMOGRAPHICS
════════════════════════════════════════════════════════════════════════════════
 
personalInfos: {
  patientName,      — Full name as printed on the report.
  age,              — Extract the age string EXACTLY as written (e.g., "21 Years", "38Y/F"). Do not alter it.
  gender,           — Male | Female | Other | null.
  dob,              — YYYY-MM-DD. Extract ONLY if explicitly printed on the document. DO NOT calculate or infer the DOB from the age. If not explicitly stated, set to null. NEVER return "".
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 3 — SCAN TECHNIQUE & CLINICAL CONTEXT
════════════════════════════════════════════════════════════════════════════════
 
scanTechnique: {
  studyType,         — Full USG study name. e.g. "USG Abdomen and Pelvis", 
                       "Obstetric Ultrasound", "Doppler Left Lower Limb".
  probeOrMethod,     — e.g. "Transabdominal", "Transvaginal (TVS)", "Linear probe",
                       "Endorectal", "Color Doppler". null if not stated.
  lmp,               — Last Menstrual Period date if stated (common in Pelvis/OB).
  gestationalAge,    — Stated gestational age by dates or scan (e.g. "12 weeks 4 days").
  indication         — Clinical reason for scan / symptoms (e.g. "Pain abdomen").
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 4 — FINDINGS BY ANATOMICAL REGION / ORGAN
════════════════════════════════════════════════════════════════════════════════
 
Extract ALL findings. Organise by the anatomical section headings present.
For Obstetric scans, use "fetalAnatomy", "placenta", "amnioticFluid", etc.
 
findings: {
  "<organOrRegionInCamelCase>": {
 
    normalFindings: [
      — Array of strings. Include EVERY statement describing something as normal,
        unremarkable, physiological, or absent of pathology.
        e.g. "Liver is normal in size and echotexture.", "No focal lesions seen."
    ],
 
    abnormalFindings: [
      — Array of objects — one per DISTINCT lesion, cyst, calculus, structural change.
      {
        site,                — Precise location (e.g. "Right lobe segment VI", "Fetal spine").
        description,         — Full verbatim finding text.
        size,                — Measurements as written (e.g. "12 x 8 mm", "1.5 cc").
        echogenicity,        — "Anechoic" | "Hypoechoic" | "Hyperechoic" | "Isoechoic" |
                               "Heterogeneous" | "Complex" | null.
        vascularity,         — Blood flow / Doppler signals if mentioned.
                               e.g. "Increased peripheral vascularity", "Avascular".
        calcification,       — true | false | null.
        morphology,          — e.g. "thin-walled cyst", "irregular solid mass", "shadowing calculus".
        changeFromPrevious   — "Increased" | "Decreased" | "Stable" | "New" | null.
      }
    ]
  }
}
 
════════════════════════════════════════════════════════════════════════════════
SECTION 5 — MEASUREMENTS & BIOMETRY SUMMARY
════════════════════════════════════════════════════════════════════════════════
 
Extract a flat list of ALL explicit measurements, especially organ sizes, volumes, 
fetal biometry (BPD, HC, AC, FL, FHR), fluid pockets, or doppler velocities.
 
measurements: [
  {
    structure,       — e.g. "Liver span", "BPD", "Fetal Heart Rate", "Right Ovary Volume", "CBD diameter".
    value,           — Numeric value + unit (e.g. "14 cm", "45 mm", "140 bpm", "12 cc").
    correspondingAge,— For fetal biometry, the age it corresponds to (e.g. "20 weeks 2 days"). null if N/A.
  }
]
 
════════════════════════════════════════════════════════════════════════════════
SECTION 6 — IMPRESSION / CONCLUSION
════════════════════════════════════════════════════════════════════════════════
 
impression: [
  — Array of strings. Each bullet point or distinct concluding sentence = one string.
  — Preserve full verbatim text. Do not summarise.
]
 
overallAssessment: — Infer ONE value:
  "Normal Study"
  "Benign Finding"
  "Suspicious / Indeterminate"
  "Malignant / Highly Suspicious"
  "Obstetric - Live Intrauterine Pregnancy"
  "Obstetric - Complicated / Abnormal"
  "Inflammatory / Infectious"
  null
 
════════════════════════════════════════════════════════════════════════════════
SECTION 7 — SIGNATORIES
════════════════════════════════════════════════════════════════════════════════
 
signatories: [
  {
    name,               — Title + full name as written.
    designation,        — e.g. "Consultant Radiologist", "Sonologist".
    degrees,            — e.g. "MD, DMRD".
  }
]
 
════════════════════════════════════════════════════════════════════════════════
OUTPUT JSON STRUCTURE
════════════════════════════════════════════════════════════════════════════════
 
{
  "documents": [
    {
      "documentName": "Ultrasound Scan",
      "documentDate": "DD MMM, YYYY",
      "reportDate": null,
      "hospitalName": null,
      "hospitalId": "",
      "referredBy": "",
 
      "personalInfos": {
        "patientName": "",
        "age": "",
        "gender": "",
        "dob": null
      },
 
      "scanTechnique": {
        "studyType": "",
        "probeOrMethod": null,
        "lmp": null,
        "gestationalAge": null,
        "indication": null
      },
 
      "findings": {},
      "measurements": [],
      
      "impression": [],
      "overallAssessment": null,
      "signatories": [],
      
      "extractedTables": [
        {
          "tableName": "Table 1 (or use the printed table heading if available)",
          "columns": ["Column 1 Name", "Column 2 Name"],
          "rows": [
            {
              "Column 1 Name": "row 1 data",
              "Column 2 Name": "row 1 data"
            }
          ]
        }
      ]
    }
  ],
  "parsedDocumentPercentage": 0
}
 
FINAL RULES:
1. NEVER alter any numeric value.
2. In Fetal/Obstetric scans, ensure FHR (Fetal Heart Rate) and biometry (BPD, HC, AC, FL) are cleanly listed in the `measurements` array.
3. Treat HTML table tags (<table>, <tr>, <td>) strictly as structured data and extract them row-by-row into `extractedTables`.
4. Output ONLY the JSON object.
"""

PROMPT_MAP = {
    "radiotherapy_report":    SYSTEM_PROMPT_RADIOTHERAPY,   
    "discharge_summary":      SYSTEM_PROMPT_DISCHARGE,
    "outpatient_note":        SYSTEM_PROMPT_OPD,
    "chemotherapy_admission": SYSTEM_PROMPT_CHEMO,
    "pet_ct_scan":            SYSTEM_PROMPT_PET_CT,
    "ct_scan":                SYSTEM_PROMPT_CT_SCAN,
    "mri_scan":               SYSTEM_PROMPT_MRI,
    "mammogram":              SYSTEM_PROMPT_MAMMOGRAM,
    "dna_test":               SYSTEM_PROMPT_DNA_TEST,
    "histopathology_report":  SYSTEM_PROMPT_HISTOPATHOLOGY,
    "cytology_report":        SYSTEM_PROMPT_HISTOPATHOLOGY,
    "ultrasound_scan":        SYSTEM_PROMPT_ULTRASOUND,
    "referral_letter":        SYSTEM_PROMPT_GENERAL,
    "registration_receipt":   SYSTEM_PROMPT_GENERAL,
}
 
USER_PROMPT = """Extract every piece of information from the OCR text below into JSON.
 
=========================================
CRITICAL EXTRACTION RULES (STRICTLY ENFORCED)
=========================================
1. ZERO HALLUCINATION POLICY: You MUST NOT guess, fabricate, or infer any data. If a specific piece of information is not explicitly written in the document, you MUST set its value to `null`. Do NOT calculate Date of Birth from age.
2. MULTI-PAGE TABULAR DATA: Medical documents often have tables that span across multiple pages. You MUST carefully track these and seamlessly merge rows from all pages into a single, continuous array in the JSON. DO NOT stop extracting a table just because you hit a "--- PAGE X ---" marker.
3. HTML TABLE PARSING: The OCR text may contain HTML table tags (<table>, <tr>, <td>). Treat these strictly as structured data. Extract every row accurately, ensuring no column data is merged incorrectly or left behind.
 
Before reading, strip: all <|ref|>…<|/ref|> tags, all <|det|>…<|/det|> tags, and all bounding box coordinates like [[x, y, x, y]]. These are OCR artifacts, not document content.
 
DOCUMENT:
{combined_text}
"""
 
def extract_json_from_response(content: str) -> str:
    content = content.strip()
    for fence in ("```json", "```"):
        if content.startswith(fence):
            content = content[len(fence):]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    start = content.find("{")
    end   = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]
    return content
  
def _call_gemini(prompt_text: str, system_msg: str, retries: int = 3, force_json: bool = True, response_schema=None, keep_alive: str = "0", num_ctx: int = 8192,        # ← add this
                 num_predict: int = 8000) -> str:
    import urllib.request
    import json
    import time

    print("\n" + "="*50)
    print("🔒🔒🔒 LOCAL OLLAMA INFERENCE TRIGGERED 🔒🔒🔒")
    print("="*50 + "\n")

    url = "http://localhost:11434/api/chat"
    model_name = "qwen3:8b" 

    json_reminder = "\n\nIMPORTANT: Return ONLY a valid JSON object. No markdown fences, no preamble." if force_json else ""
    system_content = (system_msg or "You are a medical data extractor.") + json_reminder

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": prompt_text}
    ]

    payload_dict = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive, # Dynamic memory control
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": num_ctx
        }
    }

    # Force Ollama's native JSON mode
    if force_json:
        payload_dict["format"] = "json"

    payload = json.dumps(payload_dict).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            
            print(f"  [Attempt {attempt}] Processing {len(prompt_text)} chars with local {model_name}...")
            
            # Start timer to track your local speed
            start_time = time.perf_counter()
            
            # 15-minute timeout to ensure large docs don't get cut off
            with urllib.request.urlopen(req, timeout=900) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            
            elapsed_time = time.perf_counter() - start_time
            result = body["message"]["content"].strip()
            
            print(f"  [SUCCESS] Local JSON generated in {elapsed_time:.2f} seconds!")
            return result
            
        except Exception as e:
            last_err = e
            err_msg = str(e)
            if hasattr(e, "read"):
                try:
                    err_msg += f" | Details: {e.read().decode('utf-8')}"
                except Exception:
                    pass
            wait = min(2 ** attempt, 30)
            print(f"  Local Ollama attempt {attempt}/{retries} failed: {err_msg}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Local Ollama ({model_name}) failed after {retries} attempts: {last_err}")
  
def extract_pages_by_number(combined_text: str, start_page: int, end_page: int) -> str:
    extracted = []
    pages = re.split(r'--- PAGE (\d+) ---', combined_text)
    
    for i in range(1, len(pages), 2):
        try:
            page_num = int(pages[i])
            content = pages[i+1]
            if start_page <= page_num <= end_page:
                extracted.append(f"--- PAGE {page_num} ---\n{content.strip()}")
        except (ValueError, IndexError):
            continue
            
    if not extracted:
        return combined_text 
    return "\n\n".join(extracted)
  
def _valid_document_types() -> List[str]:
    return list(PROMPT_MAP.keys()) + ["other"]

def _normalize_doc_type(raw: str) -> str:
    valid = _valid_document_types()
    detected = str(raw or "other").strip().lower().replace(" ", "_")
    if detected in valid:
        return detected
    matched = next((t for t in valid if t in detected or detected in t), None)
    return matched or "other"

def _infer_last_page(combined_text: str) -> int:
    pages = [int(m) for m in re.findall(r"--- PAGE (\d+) ---", combined_text)]
    return max(pages) if pages else 999

def _normalize_doc_segments(segments: list, combined_text: str) -> List[dict]:
    last_page = _infer_last_page(combined_text)
    normalized: List[dict] = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        start = max(1, int(seg.get("start_page", 1) or 1))
        end = int(seg.get("end_page", last_page) or last_page)
        if end < start:
            start, end = end, start
        normalized.append({
            "document_type": _normalize_doc_type(seg.get("document_type")),
            "start_page": start,
            "end_page": min(end, last_page),
        })
    if not normalized:
        normalized = [{"document_type": "other", "start_page": 1, "end_page": last_page}]
    return normalized

def _extract_document_segments(combined: str, doc_segments: List[dict]) -> dict:
    """Run type-specific extraction for each page-bounded segment SEQUENTIALLY."""
    logger.info("  Extracting %d document segment(s) with type-specific prompts …", len(doc_segments))
    for idx, doc_info in enumerate(doc_segments, 1):
        logger.info(
            "    %d. %s (pages %s–%s)",
            idx,
            doc_info.get("document_type", "other").upper(),
            doc_info.get("start_page", "?"),
            doc_info.get("end_page", "?"),
        )

    def _extract_one_doc(doc: dict):
        detected_type = doc.get("document_type", "other")
        start = int(doc.get("start_page", 1))
        end = int(doc.get("end_page", 999))
        targeted_text = extract_pages_by_number(combined, start, end)
        system_prompt = PROMPT_MAP.get(detected_type, SYSTEM_PROMPT_GENERAL)
        user_prompt = USER_PROMPT.format(combined_text=targeted_text)
        try:
            raw = _call_gemini(user_prompt, system_prompt, force_json=True,num_ctx=8192,num_predict=4096)
            extracted = json_repair.loads(raw)
            if isinstance(extracted, list):
                extracted = {"documents": extracted, "parsedDocumentPercentage": 100}
            elif not isinstance(extracted, dict):
                extracted = {"documents": [extracted], "parsedDocumentPercentage": 0}
            return extracted.get("documents", [extracted]), extracted.get("parsedDocumentPercentage", 100)
        except Exception as e:
            logger.error("  Extraction failed for %s (pages %s–%s): %s", detected_type, start, end, e)
            return [], 0

    # <--- CRITICAL FIX: Removed ThreadPoolExecutor to prevent RAM crash --->
    # Running multiple 7B models at the same time instantly triggers swapping and timeouts!
    results = []
    for doc in doc_segments:
        results.append(_extract_one_doc(doc))

    all_documents: List[dict] = []
    total_score = 0
    score_count = 0
    for docs, score in results:
        all_documents.extend(docs)
        total_score += score
        score_count += 1

    if not all_documents:
        return {"documents": [], "parsedDocumentPercentage": 0}

    parsed_pct = round(total_score / score_count) if score_count > 0 else 100
    if len(doc_segments) > 1:
        logger.info("  Multi-document extraction complete. Average score: %s%%", parsed_pct)
    else:
        logger.info("  Single-document extraction complete. Score: %s%%", parsed_pct)
    return {"documents": all_documents, "parsedDocumentPercentage": parsed_pct}

def texts_to_json(
    combined: str,
    doc_type: str = None,
    multi_doc: bool = False,
    doc_segments: List[dict] = None,
) -> dict:
    # Ensure LightOnOCR is evicted before extraction starts, so it isn't
    # still resident in VRAM (within its 5m keep_alive window) when qwen3:8b
    # loads with keep_alive=-1. Cheap no-op if it's already unloaded.
    _unload_model(OLLAMA_OCR_MODEL)

    if doc_segments:
        return _extract_document_segments(combined, doc_segments)

    if not doc_type:
        doc_type = "other"

    if multi_doc:
        logger.info("  Multi-document mode (legacy). Running page-aware classifier …")
        raw_classifier = _call_gemini(
            prompt_text=CLASSIFIER_PROMPT.replace("{combined_text}", combined),
            system_msg="",
            force_json=True,
        )
        try:
            parsed = json_repair.loads(raw_classifier)
            doc_segments = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            doc_segments = [{"document_type": doc_type, "start_page": 1, "end_page": 999}]
        doc_segments = _normalize_doc_segments(doc_segments, combined)
        return _extract_document_segments(combined, doc_segments)

    logger.info("  Single-document mode. Using prompt for: '%s'", doc_type)
    return _extract_document_segments(
        combined,
        [{"document_type": doc_type, "start_page": 1, "end_page": _infer_last_page(combined)}],
    )
 
# ============================================================================
# 10. WORKER PROCESSES
# ============================================================================
def assess_pdf_quality(pdf_path: str) -> dict:
    local_logger = get_configured_logger("QualityCheck")
    engine_override = normalize_ocr_engine(OCR_ENGINE)

    if engine_override in ("paddleocr", "lighton"):
        local_logger.info(
            "  OCR engine manually overridden to: %s", ocr_engine_label(engine_override)
        )
        return {
            "engine": engine_override,
            "quality": "good" if engine_override == "paddleocr" else "poor",
            "selectable_ratio": -1.0,
            "pages_sampled": 0,
            "reason": f"Manual override: OCR_ENGINE='{engine_override}'",
        }

    if not _PADDLE_AVAILABLE:
        local_logger.warning(
            "  PaddleOCR not installed — forcing LightOnOCR regardless of quality. "
            "Install with: pip install paddlepaddle-gpu paddleocr"
        )
        return {
            "engine": "lighton",
            "quality": "unknown",
            "selectable_ratio": -1.0,
            "pages_sampled": 0,
            "reason": "PaddleOCR not installed",
        }

    local_logger.info("  Assessing PDF text-layer quality …")
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        total = len(pdf)
        sample_indices = list(range(min(total, 10))) 
        selectable_count = 0

        for idx in sample_indices:
            page = pdf[idx]
            try:
                text_page = page.get_textpage()
                text = text_page.get_text_range()
                printable = sum(1 for c in text if c.isprintable() and not c.isspace())
                if printable >= MIN_CHARS_PER_PAGE:
                    selectable_count += 1
                local_logger.info(
                    f"    Page {idx + 1}: {printable} printable chars "
                    f"→ {'selectable' if printable >= MIN_CHARS_PER_PAGE else 'image-only'}"
                )
            except Exception:
                local_logger.info(f"    Page {idx + 1}: text-layer extraction failed → image-only")

        pdf.close()

        ratio = selectable_count / len(sample_indices)
        is_good = ratio >= SELECTABLE_TEXT_RATIO
        engine = "paddleocr" if is_good else "lighton"
        quality = "good" if is_good else "poor"
        reason = (
            f"{selectable_count}/{len(sample_indices)} sampled pages are selectable "
            f"(ratio={ratio:.2f}, threshold={SELECTABLE_TEXT_RATIO})"
        )

        # If it has no text layer, do a quick PaddleOCR test
        if not is_good and _PADDLE_AVAILABLE:
            try:
                local_logger.info("  Scanned PDF detected. Running quality check using PaddleOCR on Page 1...")
                page1_img = os.path.join(PDF_WORKSPACE_DIR, "page_1.png")
                if not os.path.exists(page1_img):
                    pdf_obj = pdfium.PdfDocument(pdf_path)
                    page = pdf_obj[0]
                    bitmap = page.render(scale=RENDER_SCALE)
                    img = bitmap.to_pil().convert("RGB")
                    os.makedirs(PDF_WORKSPACE_DIR, exist_ok=True)
                    img.save(page1_img)
                    pdf_obj.close()

                api_ver = _detect_paddle_api_version()
                use_gpu = PADDLE_USE_GPU
                import tempfile
                import subprocess
                worker_fd, worker_path = tempfile.mkstemp(suffix="_paddle_quality_worker.py", prefix="med_")
                try:
                    with os.fdopen(worker_fd, "w", encoding="utf-8") as f:
                        f.write(_PADDLE_WORKER_SCRIPT)
                    config = {
                        "img_dir":     PDF_WORKSPACE_DIR,
                        "out_dir":     PDF_WORKSPACE_DIR,
                        "total_pages": 1,
                        "use_gpu":     use_gpu,
                        "api_ver":     api_ver,
                        "force_rerun": True,
                    }
                    paddle_env = os.environ.copy()
                    paddle_env["PYTHONIOENCODING"] = "utf-8"
                    paddle_env["PYTHONUTF8"] = "1"
                    proc = subprocess.run(
                        [sys.executable, worker_path],
                        input=json.dumps(config).encode("utf-8"),
                        capture_output=True,
                        timeout=90,
                        env=paddle_env,
                    )
                    stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace")
                    
                    avg_conf = 0.0
                    for line in stderr_text.splitlines():
                        if "Page 1 average confidence:" in line:
                            try:
                                avg_conf = float(line.split("Page 1 average confidence:")[-1].strip())
                            except Exception:
                                pass
                    
                    local_logger.info(f"  PaddleOCR Page 1 average confidence: {avg_conf:.4f}")
                    
                    # <--- CRITICAL FIX: Changed from 0.99 to 0.75 --->
                    # 0.99 is impossible for medical scans. 0.75 properly captures decent scans.
                    if avg_conf >= 0.75:
                        engine = "paddleocr"
                        quality = "scanned_good"
                        reason += f" | Quality check: Page 1 confidence {avg_conf:.2f} >= 0.75 (good scan) -> using fast PaddleOCR."
                    else:
                        engine = "lighton"
                        quality = "scanned_poor"
                        reason += f" | Quality check: Page 1 confidence {avg_conf:.2f} < 0.75 (poor scan) -> fallback to LightOnOCR."
                finally:
                    try:
                        os.remove(worker_path)
                    except OSError:
                        pass
            except Exception as e:
                local_logger.warning(f"  PaddleOCR quality check failed: {e}. Defaulting to lighton.")

        local_logger.info(
            "  Quality assessment → %s | Engine → %s | %s",
            quality.upper(),
            ocr_engine_label(engine),
            reason,
        )
        return {
            "engine": engine,
            "quality": quality,
            "selectable_ratio": ratio,
            "pages_sampled": len(sample_indices),
            "reason": reason,
        }

    except Exception as e:
        local_logger.error(
            "  Quality assessment failed (%s). Defaulting to LightOnOCR.", e
        )
        return {
            "engine": "lighton",
            "quality": "unknown",
            "selectable_ratio": -1.0,
            "pages_sampled": 0,
            "reason": f"Assessment error: {e}",
        }


def extract_pdf_text_pages(pdf_path: str) -> List[Tuple[int, str]]:
    page_texts: List[Tuple[int, str]] = []
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        for i in range(len(pdf)):
            page_num = i + 1
            text = ""
            try:
                text_page = pdf[i].get_textpage()
                text = text_page.get_text_range() or ""
            except Exception:
                pass
            page_texts.append((page_num, text.strip()))
        pdf.close()
    except Exception as e:
        logger.error(f"  PDF text-layer extraction failed: {e}")
    return page_texts

_PADDLE_WORKER_SCRIPT = r"""
# paddle_worker.py — spawned as a fresh subprocess by medical_agent.py
# Supports PaddleOCR v2 (.ocr), v3 (.predict), and the v3 new-style result object.
import sys, json, os, io

# Force UTF-8 on stdio so the parent process never hits cp1252 decode errors on Windows.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

cfg          = json.loads(sys.stdin.read())
img_dir      = cfg["img_dir"]
out_dir      = cfg["out_dir"]
total_pages  = cfg["total_pages"]
use_gpu      = cfg["use_gpu"]
force_rerun  = cfg["force_rerun"]

import logging
logging.disable(logging.WARNING)
os.environ["GLOG_minloglevel"] = "3"

results = {}

from paddleocr import PaddleOCR


# ── Detect API generation once at startup ─────────────────────────────────────
def _detect_api_gen():
    # Return 'v3' if PaddleOCR >= 3.x is installed, else 'v2'.
    try:
        import paddleocr as _poc
        ver = getattr(_poc, "__version__", "") or ""
        major = int(ver.split(".")[0]) if ver and ver[0].isdigit() else 0
        if major >= 3:
            return "v3"
        # Secondary check: v3 ships _pipelines subpackage
        import importlib
        if importlib.util.find_spec("paddleocr._pipelines") is not None:
            return "v3"
    except Exception:
        pass
    return "v2"

API_GEN = _detect_api_gen()
sys.stderr.write(f"[paddle_worker] Detected PaddleOCR API generation: {API_GEN}\n")


# ── Engine factory ────────────────────────────────────────────────────────────
def get_ocr_engine(try_gpu=True):
    device = "gpu:0" if try_gpu else "cpu"
    v3_attempts = [
        {
            "lang": "en",
            "device": device,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
        },
        {"lang": "en", "device": device},
        {"lang": "en"},
    ]
    v2_attempts = [
        {"use_angle_cls": True, "lang": "en", "use_gpu": try_gpu},
        {"lang": "en", "use_gpu": try_gpu},
        {"lang": "en"},
    ]

    attempts = v3_attempts if API_GEN == "v3" else v2_attempts
    last_err = None
    for kwargs in attempts:
        try:
            engine = PaddleOCR(**kwargs)
            sys.stderr.write(f"[paddle_worker] engine ready ({kwargs})\n")
            return engine
        except Exception as e:
            last_err = e
            sys.stderr.write(f"[paddle_worker] init failed ({kwargs}): {e}\n")

    raise RuntimeError(f"All PaddleOCR init attempts failed. Last error: {last_err}")


# ── Result parsers ────────────────────────────────────────────────────────────
def _lines_from_ocr_item(item, confs=None):
    # PaddleOCR 3.x OCRResult is a dict subclass with rec_texts / rec_scores.
    lines = []
    if item is None:
        return lines
    texts = scores = None
    if isinstance(item, dict):
        texts = item.get("rec_texts") or []
        scores = item.get("rec_scores") or []
    elif hasattr(item, "get"):
        texts = item.get("rec_texts") or []
        scores = item.get("rec_scores") or []
    if texts is not None:
        if not scores:
            scores = [1.0] * len(texts)
        for txt, score in zip(texts, scores):
            if isinstance(score, (int, float)) and confs is not None:
                confs.append(float(score))
            if float(score) >= 0.5 and txt and str(txt).strip():
                lines.append(str(txt).strip())
        return lines
    # Legacy per-line dict rows
    if isinstance(item, dict):
        txt = item.get("rec_text", "") or ""
        score = item.get("rec_score", 0.0) or 0.0
        if isinstance(score, (int, float)) and confs is not None:
            confs.append(float(score))
        if float(score) >= 0.5 and str(txt).strip():
            lines.append(str(txt).strip())
    return lines


def _parse_v3_result(result, confs=None):
    lines = []
    if result is None:
        return lines
    items = result if isinstance(result, (list, tuple)) else [result]
    for item in items:
        if isinstance(item, (list, tuple)):
            lines.extend(_parse_v2_result([item], confs))
        else:
            lines.extend(_lines_from_ocr_item(item, confs))
    return lines


def _parse_v2_result(result, confs=None):
    lines = []
    if result is None or result == [None]:
        return lines
    pages = result if isinstance(result, (list, tuple)) else [result]
    for page in pages:
        if not page:
            continue
        items = page if isinstance(page, (list, tuple)) else [page]
        for item in items:
            if not item or not isinstance(item, (list, tuple)):
                continue
            if len(item) >= 2:
                label = item[1]
                if isinstance(label, (list, tuple)) and len(label) >= 2:
                    txt, conf = label[0], label[1]
                    if isinstance(conf, (int, float)) and confs is not None:
                        confs.append(float(conf))
                    if isinstance(conf, (int, float)) and conf >= 0.5 and txt and str(txt).strip():
                        lines.append(str(txt).strip())
                elif isinstance(label, str) and label.strip():
                    lines.append(label.strip())
    return lines


def parse_result(result, confs=None):
    # Dispatch to the correct parser based on detected API generation.
    if API_GEN == "v3":
        lines = _parse_v3_result(result, confs)
        # If v3 parser found nothing, the engine may have returned v2-style data
        if not lines:
            lines = _parse_v2_result(result, confs)
        return lines
    return _parse_v2_result(result, confs)


# ── Core inference wrapper ────────────────────────────────────────────────────
def run_ocr_on_image(engine, img_path, confs=None):
    # v3: predict(); v2: ocr()
    if hasattr(engine, "predict"):
        res = engine.predict(img_path)
    else:
        res = engine.ocr(img_path, cls=True)

    if res is None or res == [None] or res == []:
        raise RuntimeError("PaddleOCR returned empty/None result.")

    lines = parse_result(res, confs)
    if not lines:
        sys.stderr.write(
            f"[paddle_worker] parse returned 0 lines for {img_path}; "
            f"result_len={len(res) if res else 0}\n"
        )
    return "\n".join(lines)


# ── Initialise engine ─────────────────────────────────────────────────────────
try:
    ocr = get_ocr_engine(try_gpu=use_gpu)
except Exception as init_err:
    sys.stderr.write(f"[paddle_worker] FATAL init error: {init_err}\n")
    print(json.dumps({str(i): "" for i in range(1, total_pages + 1)}))
    sys.exit(1)
gpu_failed_permanently = False  # once True, all remaining pages use CPU


# ── Per-page loop ─────────────────────────────────────────────────────────────
for page_num in range(1, total_pages + 1):
    img_path = os.path.join(img_dir, f"page_{page_num}.png")
    out_path = os.path.join(out_dir, f"raw_text_page_{page_num}.txt")

    # Cache hit — only accept non-empty files
    if not force_rerun and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as fh:
            cached = fh.read()
        if cached.strip():
            results[str(page_num)] = cached
            continue
        else:
            sys.stderr.write(
                f"[paddle_worker] Page {page_num}: stale empty cache — re-running OCR.\n"
            )
            try:
                os.remove(out_path)
            except OSError:
                pass

    if not os.path.exists(img_path):
        sys.stderr.write(f"[paddle_worker] Page {page_num}: image not found — skipping.\n")
        results[str(page_num)] = ""
        continue

    text = ""
    confs = []

    # GPU attempt
    if use_gpu and not gpu_failed_permanently:
        try:
            text = run_ocr_on_image(ocr, img_path, confs)
        except Exception as gpu_err:
            sys.stderr.write(
                f"[paddle_worker] Page {page_num} GPU failure: {gpu_err}\n"
                f"[paddle_worker] Switching to CPU engine permanently.\n"
            )
            gpu_failed_permanently = True
            ocr = get_ocr_engine(try_gpu=False)
            text = ""
            confs = []

    # CPU retry — also when GPU returned empty text without raising
    if not text.strip():
        if use_gpu and not gpu_failed_permanently:
            sys.stderr.write(
                f"[paddle_worker] Page {page_num}: empty GPU result — retrying on CPU.\n"
            )
            gpu_failed_permanently = True
            ocr = get_ocr_engine(try_gpu=False)
        try:
            text = run_ocr_on_image(ocr, img_path, confs)
        except Exception as cpu_err:
            sys.stderr.write(
                f"[paddle_worker] Page {page_num} CPU inference failed: {cpu_err}\n"
            )
            text = ""
            confs = []

    # NEVER write an empty cache file
    if text.strip():
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        sys.stderr.write(f"[paddle_worker] Page {page_num} average confidence: {avg_conf:.4f}\n")
        sys.stderr.write(f"[paddle_worker] Page {page_num}: {len(text)} chars written.\n")
    else:
        sys.stderr.write(
            f"[paddle_worker] Page {page_num}: empty result — cache NOT written.\n"
        )

    results[str(page_num)] = text

print(json.dumps(results))
"""


def _detect_paddle_api_version() -> str:
    try:
        import paddleocr as _poc
        ver = getattr(_poc, "__version__", "") or ""
        major = int(ver.split(".")[0]) if ver and ver[0].isdigit() else 0
        if major >= 3:
            return "v3"
        import importlib
        if importlib.util.find_spec("paddleocr._pipelines") is not None:
            return "v3"
    except Exception:
        pass
    return "v2"


def transcribe_all_pages_paddle(
    total_pages: int,
    compression_mode: str = "large",
    force_rerun: bool = False,
    workspace_dir: str = None,
) -> List[Tuple[int, str]]:
    import subprocess
    import tempfile

    ws = workspace_dir or PDF_WORKSPACE_DIR

    api_ver = _detect_paddle_api_version()
    use_gpu = PADDLE_USE_GPU

    logger.info(f"  Launching isolated PaddleOCR subprocess (api={api_ver}, use_gpu={use_gpu}) …")

    worker_fd, worker_path = tempfile.mkstemp(suffix="_paddle_worker.py", prefix="med_")
    try:
        with os.fdopen(worker_fd, "w", encoding="utf-8") as f:
            f.write(_PADDLE_WORKER_SCRIPT)

        config = {
            "img_dir":     ws,
            "out_dir":     ws,
            "total_pages": total_pages,
            "use_gpu":     use_gpu,
            "api_ver":     api_ver,
            "force_rerun": force_rerun,
        }
        config_json = json.dumps(config)

        paddle_env = os.environ.copy()
        paddle_env["PYTHONIOENCODING"] = "utf-8"
        paddle_env["PYTHONUTF8"] = "1"
        paddle_env["FLAGS_allocator_strategy"] = "auto_growth"

        proc = subprocess.run(
            [sys.executable, worker_path],
            input=config_json.encode("utf-8"),
            capture_output=True,
            timeout=600,
            env=paddle_env,
        )

        stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace")

        if proc.returncode != 0:
            logger.error(
                f"  PaddleOCR subprocess failed (exit {proc.returncode}):\n{stderr_text[-2000:]}"
            )
            return []

        raw_out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        if stderr_text.strip():
            logger.warning(f"  PaddleOCR Subprocess Logs/Errors:\n{stderr_text.strip()}")
        json_start = raw_out.rfind("{")
        if json_start == -1:
            logger.error("  PaddleOCR subprocess produced no JSON output.")
            logger.debug(f"  stdout was: {raw_out[:500]}")
            return []

        results_map = json.loads(raw_out[json_start:])

    except subprocess.TimeoutExpired:
        logger.error("  PaddleOCR subprocess timed out (>600 s).")
        return []
    except Exception as e:
        logger.error(f"  PaddleOCR subprocess error: {e}")
        return []
    finally:
        try:
            os.unlink(worker_path)
        except Exception:
            pass

    page_texts: List[Tuple[int, str]] = []
    for page_num in range(1, total_pages + 1):
        text = results_map.get(str(page_num), "")
        if text:
            logger.info(f"  Page {page_num}: {len(text)} chars via PaddleOCR.")
        else:
            logger.warning(f"  Page {page_num}: no text from PaddleOCR.")
        page_texts.append((page_num, text))

    return page_texts

def run_ocr_phase(
    enhance_contrast_flag: bool,
    force_rerun: bool = False,
    compression_mode: str = "large",
    pdf_path: str = None,
    workspace_dir: str = None,
    stitched_path: str = None,
):
    local_logger = get_configured_logger("OCR-Phase")
    local_logger.info("\n=== STAGE 1: PDF → OCR ===")

    _pdf_path      = pdf_path      or PDF_PATH
    _workspace_dir = workspace_dir or PDF_WORKSPACE_DIR
    _stitched_path = stitched_path or STITCHED_PATH

    if os.path.exists(_stitched_path) and not force_rerun:
        local_logger.info(f"  {_stitched_path} already exists — skipping OCR. Delete it to re-run.")
        return

    quality_info = assess_pdf_quality(_pdf_path)
    engine       = quality_info["engine"]
    local_logger.info(
        "  → OCR engine selected: %s  (quality=%s, reason: %s)",
        ocr_engine_label(engine),
        quality_info["quality"],
        quality_info["reason"],
    )

    total_pages = pdf_to_images(_pdf_path, RENDER_SCALE, enhance_contrast_flag)
    if not total_pages:
        local_logger.error("  No pages rendered. Aborting.")
        return

    if engine == "paddleocr":
        local_logger.info("  Extracting embedded PDF text layer (fast path for digital PDF) …")
        page_texts = extract_pdf_text_pages(_pdf_path)
        valid = [(n, t.strip()) for n, t in page_texts if t and t.strip()]
        min_pages = max(1, int(total_pages * SELECTABLE_TEXT_RATIO))

        if len(valid) >= min_pages:
            combined = "\n\n".join(f"--- PAGE {n} ---\n{t}" for n, t in valid)
            with open(_stitched_path, "w", encoding="utf-8") as f:
                f.write(combined)
            local_logger.info(
                f"  Text layer: {len(valid)}/{total_pages} page(s) → "
                f"{len(combined)} chars → {_stitched_path}"
            )
            local_logger.info("=== STAGE 1 COMPLETE (PDF text layer) ===\n")
            return

        local_logger.warning(
            f"  Embedded text only on {len(valid)}/{total_pages} pages "
            f"(need ≥{min_pages}). Running PaddleOCR on page images …"
        )
        page_texts = transcribe_all_pages_paddle(
            total_pages,
            compression_mode=compression_mode,
            force_rerun=force_rerun,
            workspace_dir=_workspace_dir,
        )
        valid = [(n, t.strip()) for n, t in page_texts if t and t.strip()]
        if not valid:
            local_logger.warning("  PaddleOCR produced no output. Falling back to LightOnOCR via Ollama …")
            engine = "lighton"
        else:
            combined = "\n\n".join(f"--- PAGE {n} ---\n{t}" for n, t in valid)
            with open(_stitched_path, "w", encoding="utf-8") as f:
                f.write(combined)
            local_logger.info(
                f"  PaddleOCR stitched {len(valid)} page(s) → {len(combined)} chars → {_stitched_path}"
            )
            local_logger.info("=== STAGE 1 COMPLETE (PaddleOCR) ===\n")
            return

    local_logger.info("  Running LightOnOCR via Ollama (high-accuracy path for poor-quality PDF) …")
    ocr_model, ocr_tokenizer = load_ocr_model()
    page_texts = transcribe_all_pages(
        total_pages, ocr_model, ocr_tokenizer,
        compression_mode,
        force_rerun=force_rerun,
    )
    stitch_pages(page_texts, stitched_path=_stitched_path)
    free_memory(ocr_model, ocr_tokenizer)
    local_logger.info("=== STAGE 1 COMPLETE (LightOnOCR) ===\n")

# ============================================================================
# 10.5  AUTO DOCUMENT STRUCTURE CLASSIFIER  (single vs multi + page ranges)
# ============================================================================
_STRUCTURE_CLASSIFIER_PROMPT = """You are a medical document structure analyzer.
The text below is raw OCR output from a medical PDF, divided by --- PAGE X --- markers.

STEP 1 — Decide if this PDF is ONE document or MULTIPLE distinct documents.
STEP 2 — For each distinct document, identify its type and exact page range.

A PDF is MULTIPLE when it contains separate reports merged together, e.g.:
  discharge summary (pages 1–3) + histopathology (pages 4–5) + PET-CT (pages 6–8).

A PDF is SINGLE when the entire content is one report type (even if long).

Return ONLY a JSON object (no markdown, no explanation):
{{
  "mode": "single" | "multiple",
  "documents": [
    {{
      "document_type": "<type>",
      "start_page": <int>,
      "end_page": <int>
    }}
  ]
}}

document_type MUST be exactly one of:
{valid_types}

Type hints:
- "DISCHARGE SUMMARY", admission/discharge dates → discharge_summary
- Chemo cycle, infusion, regimen → chemotherapy_admission
- Radiation, fractions, Gy, DVH → radiotherapy_report
- PET, PET-CT, SUV → pet_ct_scan
- CT / CECT / HRCT (no PET) → ct_scan
- MRI, T1/T2/FLAIR → mri_scan
- Mammography, BI-RADS → mammogram
- Histopathology, biopsy, HPE, gross description → histopathology_report
- FNAC, cytology smear → cytology_report
- Ultrasound, USG, sonography → ultrasound_scan
- OPD / clinic / follow-up note → outpatient_note
- DNA / NGS / mutation panel → dna_test
- Referral letter → referral_letter
- Bill / receipt → registration_receipt
- None of the above → other

Rules:
- Page ranges must be contiguous and non-overlapping; cover every substantive page.
- For SINGLE mode, "documents" has exactly one entry spanning all pages.
- For MULTIPLE mode, "documents" has 2+ entries with different document_type values.

DOCUMENT:
{combined_text}
"""

def analyze_document_structure(combined_text: str) -> dict:
    valid_types_str = "\n".join(f'  - "{t}"' for t in _valid_document_types())
    text_for_llm = combined_text
    if len(combined_text) > 100_000:
        text_for_llm = (
            combined_text[:50_000]
            + "\n\n...[middle truncated for classifier]...\n\n"
            + combined_text[-50_000:]
        )

    prompt = _STRUCTURE_CLASSIFIER_PROMPT.format(
        valid_types=valid_types_str,
        combined_text=text_for_llm,
    )

    try:
        raw = _call_gemini(prompt, system_msg="", force_json=True)
        logger.info("  Structure classifier raw response: %s", raw[:300])
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(raw)

        segments = _normalize_doc_segments(parsed.get("documents", []), combined_text)
        mode = str(parsed.get("mode", "")).strip().lower()
        if mode not in ("single", "multiple"):
            mode = "multiple" if len(segments) > 1 else "single"
        if len(segments) == 1:
            mode = "single"
        elif len(segments) > 1:
            types = {s["document_type"] for s in segments}
            mode = "multiple" if len(types) > 1 else "single"

        return {
            "mode": mode,
            "documents": segments,
            "primary_type": segments[0]["document_type"],
        }
    except Exception as e:
        logger.error("  Document structure analysis failed (%s). Defaulting to single/other.", e)
        last_page = _infer_last_page(combined_text)
        return {
            "mode": "single",
            "documents": [{"document_type": "other", "start_page": 1, "end_page": last_page}],
            "primary_type": "other",
        }


def auto_detect_document_type(stitched_text: str) -> str:
    return analyze_document_structure(stitched_text)["primary_type"]

import json

def chunk_document_text(text: str, max_chars: int = 5000) -> list[str]:
    """
    Splits a large document into smaller chunks to prevent LLM memory thrashing.
    max_chars=5000 is roughly 1000-1200 tokens, a sweet spot for 8B models.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        # +1 accounts for the space character
        if current_length + len(word) + 1 > max_chars:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def deep_merge_json(base_dict: dict, new_dict: dict) -> dict:
    """
    Recursively merges extracted JSON chunks so lists (like medications)
    are appended rather than overwritten.
    """
    for key, value in new_dict.items():
        if key in base_dict:
            if isinstance(base_dict[key], list) and isinstance(value, list):
                base_dict[key].extend(value)
            elif isinstance(base_dict[key], dict) and isinstance(value, dict):
                base_dict[key] = deep_merge_json(base_dict[key], value)
            else:
                # If it's a string/int and already exists, keep the longest/most detailed one
                if isinstance(value, str) and isinstance(base_dict[key], str):
                    if len(value) > len(base_dict[key]):
                        base_dict[key] = value
        else:
            base_dict[key] = value
    return base_dict

def run_pipeline_extraction(*args, **kwargs) -> dict:
    """
    Orchestrates the chunking, sequential LLM extraction, and aggregation.
    Now correctly reads the raw OCR text from stitched_path.
    """
    full_document_text = ""

    # 1. Primary method: Read from the OCR stitched path.
    # Default to STITCHED_PATH if running locally via main()
    stitched_path = kwargs.get('stitched_path', STITCHED_PATH)
    if stitched_path and os.path.exists(stitched_path):
        try:
            with open(stitched_path, 'r', encoding='utf-8') as f:
                full_document_text = f.read()
            logger.info(f"Successfully loaded {len(full_document_text)} characters from {stitched_path}")
        except Exception as e:
            logger.error(f"Failed to read stitched_path {stitched_path}: {e}")

    # 2. Fallback method: Check if text was passed directly in memory
    if not full_document_text:
        text_keys = ['full_document_text', 'extracted_text', 'text', 'document_text']
        full_document_text = next((kwargs[k] for k in text_keys if k in kwargs and isinstance(kwargs[k], str)), "")

    # 3. Guardrail
    if not full_document_text:
        logger.error(f"Missing text payload! Received args: {kwargs.keys()}")
        return {"error": "Could not find document text or valid stitched_path."}

    # 4. Safely extract the document type
    document_type = kwargs.get('document_type', 'auto')
    if args and isinstance(args[0], str):
        document_type = args[0] # Handles document_type passed from main()

    # Break the document into safe sizes
    chunks = chunk_document_text(full_document_text, max_chars=4000)
    logger.info(f"Document split into {len(chunks)} chunks for processing.")
    
    final_extracted_data = {}
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1} of {len(chunks)}...")
        
        # DEFINE THE MISSING VARIABLES HERE
        system_instruction = "You are a medical data extraction system. Extract structured information. Output ONLY valid JSON."
        user_prompt = f"""
        Document Type: {document_type}
        
        If a field is not present in this specific segment, do not make it up. 
        Output ONLY valid JSON.
        
        DOCUMENT SEGMENT:
        {chunk}
        """
        
        # Call your model using the correct variable names
        response_text = _call_gemini(
            prompt_text=user_prompt, 
            system_msg=system_instruction,
            keep_alive=-1,
            num_ctx=4096,
            num_predict=2048 
        )
        
        try:
            # Safely load the JSON
            chunk_data = json_repair.loads(response_text) 
            
            # Catch instances where the LLM returns a raw list instead of a dictionary
            if isinstance(chunk_data, list):
                chunk_data = {"documents": chunk_data}
                
            # Merge the chunks
            final_extracted_data = deep_merge_json(final_extracted_data, chunk_data)
            
        except Exception as e:
            logger.error(f"Failed to parse JSON from chunk {i+1}. Skipping. Error: {e}")
            continue
            
    # --- FINAL FORMATTING FIX ---
    if not final_extracted_data:
        final_extracted_data = {"documents": [], "parsedDocumentPercentage": 0}
        
    if "documents" not in final_extracted_data:
        final_extracted_data = {
            "documents": [final_extracted_data],
            "parsedDocumentPercentage": 100
        }
    else:
        if "parsedDocumentPercentage" not in final_extracted_data:
            final_extracted_data["parsedDocumentPercentage"] = 100

    # --- THE MISSING DISK WRITE ---
    # api_server.py expects the final JSON to be physically saved to result_path!
    result_path = kwargs.get('result_path')
    
    # Fallback to local RESULT_PATH if running via terminal instead of API
    if not result_path:
        try:
            result_path = RESULT_PATH
        except NameError:
            pass
            
    if result_path:
        try:
            import json
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(final_extracted_data, f, indent=4)
            logger.info(f"Successfully saved final JSON to disk at: {result_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON to disk: {e}")

    return final_extracted_data


 
# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main() -> None:
    multiprocessing.set_start_method("spawn", force=True)

    valid_types = ["auto"] + list(PROMPT_MAP.keys()) + ["other"]
    print("\nDocument types:")
    print("  0. auto  ← let the AI detect the type automatically (recommended)")
    for i, t in enumerate(PROMPT_MAP.keys(), 1):
        print(f"  {i}. {t}")
    print(f"  {len(valid_types) - 1}. other")
    while True:
        raw = input("Select document type (number or name) [Default: auto]: ").strip().lower()
        if not raw or raw in ("0", "auto"):
            selected_type = "auto"
            break
        if raw.isdigit() and 1 <= int(raw) < len(valid_types):
            selected_type = valid_types[int(raw)]
            break
        elif raw in valid_types:
            selected_type = raw
            break
        print(f"  Invalid choice.")
 
    valid_compression_modes = ["tiny", "small", "base", "large", "high"]
    print("\nCompression modes (affects OCR accuracy and VRAM usage):")
    for i, mode in enumerate(valid_compression_modes, 1):
        print(f"  {i}. {mode}")
    while True:
        comp_raw = input("Select compression mode (number or name) [Default: large]: ").strip().lower()
        if not comp_raw:
            selected_comp = "large"
            break
        if comp_raw.isdigit() and 1 <= int(comp_raw) <= len(valid_compression_modes):
            selected_comp = valid_compression_modes[int(comp_raw) - 1]
            break
        elif comp_raw in valid_compression_modes:
            selected_comp = comp_raw
            break
        print(f"  Invalid choice. Enter a number 1-{len(valid_compression_modes)} or the exact name.")


    print("\nOCR engine selection:")
    print("  1. auto      — inspect PDF quality and decide automatically (recommended)")
    print("  2. paddleocr — force PaddleOCR (fast, best for clean/digital PDFs)")
    print("  3. lightonOCR — force LightOnOCR via Ollama (best for scanned/low-quality PDFs)")
    while True:
        engine_raw = input("Select OCR engine [Default: auto]: ").strip().lower()
        if not engine_raw or engine_raw in ("1", "auto"):
            selected_engine = "auto"
            break
        elif engine_raw in ("2", "paddleocr"):
            selected_engine = "paddleocr"
            break
        elif engine_raw in ("3", "lighton", "lightonocr", "deepseek"):
            selected_engine = "lighton"
            break
        print("  Invalid choice. Enter 1/auto, 2/paddleocr, or 3/lighton.")

    enhance_raw = input("Enhance contrast for this document? (y/n): ").strip().lower()
    selected_enhance = (enhance_raw == "y")

    rerun_raw = input("Force re-run OCR even if checkpoint exists? (y/n): ").strip().lower()
    selected_rerun = (rerun_raw == "y")
 
    global DOCUMENT_TYPE, ENHANCE_CONTRAST, COMPRESSION_MODE, OCR_ENGINE
    DOCUMENT_TYPE    = selected_type
    ENHANCE_CONTRAST = selected_enhance
    COMPRESSION_MODE = selected_comp
    OCR_ENGINE       = selected_engine
 
    logger.info(f"  Document type      : {DOCUMENT_TYPE}")
    logger.info(f"  Compression mode   : {COMPRESSION_MODE}")
    logger.info(f"  Enhance contrast   : {ENHANCE_CONTRAST}")
    logger.info(f"  Multi-document     : auto-detect (when doc type is 'auto')")
    logger.info(f"  Force OCR re-run   : {selected_rerun}")
    logger.info(f"  OCR engine         : {OCR_ENGINE}")
 
    logger.info("=== PIPELINE START ===")
 
    p1 = multiprocessing.Process(
        target=run_ocr_phase,
        args=(selected_enhance, selected_rerun, COMPRESSION_MODE),
        name="OCR-Phase"
    )
    p1.start()
    p1.join() 
 
    if p1.exitcode != 0:
        logger.error(f"OCR phase exited with code {p1.exitcode}. Aborting extraction.")
        return
 
    p2 = multiprocessing.Process(
        target=run_pipeline_extraction ,
        args=(DOCUMENT_TYPE, ENHANCE_CONTRAST, False),
        name="Extraction-Phase"
    )
    p2.start()
    p2.join()
 
    if p2.exitcode != 0:
        logger.error(f"Extraction phase exited with code {p2.exitcode}.")
    else:
        logger.info("=== PIPELINE COMPLETE ===")
 
if __name__ == "__main__":
    main()