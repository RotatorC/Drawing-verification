import re

RE_LIST_NO = re.compile(r"\b[A-Z][0-9]\s*-\s*\d{3,4}(?!\d)")  # A0-001 (뒤에 숫자 없음)
RE_DRAWING_NO = re.compile(r"\b[A-Z]\s*-\s*\d{3,4}\b")    # A-101
RE_SCALE = re.compile(r"(1\s*[:/]\s*\d+)")

def normalize_spaces(s: str) -> str:
    return " ".join((s or "").split()).strip()

def fix_common_ocr_no(s: str) -> str:
    s = (s or "")
    # AO- -> A0-
    s = s.replace("AO-", "A0-").replace("Ao-", "A0-").replace("aO-", "A0-")
    # OCR이 종종 하이픈을 다른 문자로 인식함(–, —, ㅡ 등) → '-'로 통일
    s = s.replace("–", "-").replace("—", "-").replace("-", "-").replace("ㅡ", "-")
    return s

def extract_list_no(text: str) -> str:
    t = normalize_spaces(text or "").replace(" ", "")
    t = fix_common_ocr_no(t)
    m = RE_LIST_NO.search(t)
    if not m:
        return ""
    no = m.group(0)
    no = no.replace("O", "0")  # OCR O->0
    return no

def extract_drawing_no(text: str) -> str:
    t = normalize_spaces(text).replace(" ", "")
    m = RE_DRAWING_NO.search(t)
    if not m:
        return ""
    no = m.group(0).replace(" ", "")
    prefix, nums = no.split("-")
    nums = nums.replace("O", "0").replace("o", "0")
    return f"{prefix}-{nums}"

def normalize_scale(s: str) -> str:
    s = normalize_spaces(s)
    m = RE_SCALE.search(s)
    if not m:
        return ""
    val = m.group(1).replace(" ", "").replace("/", ":")
    left, right = val.split(":")
    right = right.replace("O", "0").replace("o", "0")
    return f"{left}:{right}"

def clean_scale_from_name(name: str) -> tuple[str, str, str]:
    """
    도면명에 섞인 축척(1/400, 1/800, NONE 등) 제거하고,
    발견된 축척이 있으면 scale_a1, scale_a3로 반환.
    반환: (정제된_도면명, scale_a1, scale_a3)
    """
    s = normalize_spaces(name or "")
    scale_a1, scale_a3 = "", ""

    # "1/400 | 1/800" 또는 "1:400 1:800" 패턴 → 두 축척 추출
    m = re.search(r"(1\s*[/:]\s*\d+)\s*[/|]\s*(1\s*[/:]\s*\d+)", s)
    if m:
        scale_a1 = normalize_scale(m.group(1))
        scale_a3 = normalize_scale(m.group(2))
        s = re.sub(r"[\s|\[\]`'、。]*(1\s*[/:]\s*\d+)\s*[/|]\s*(1\s*[/:]\s*\d+)[\s|\[\]`'、。]*", " ", s)
    else:
        # 단일 축척
        m = re.search(r"(1\s*[/:]\s*\d+)", s)
        if m:
            val = normalize_scale(m.group(1))
            scale_a1 = val
            s = re.sub(r"[\s|\[\]`'、。]*(1\s*[/:]\s*\d+)[\s|\[\]`'、。]*", " ", s)

    # NONE NONE
    if re.search(r"NONE\s*[/|]?\s*NONE", s, re.I):
        if not scale_a1:
            scale_a1 = "NONE"
        if not scale_a3:
            scale_a3 = "NONE"
        s = re.sub(r"NONE\s*[/|]?\s*NONE", " ", s, flags=re.I)

    # | 1400 | 1/800 같은 OCR 노이즈
    s = re.sub(r"\|\s*1\s*[/:]?\s*\d+\s*[/|]?\s*(?:1\s*[/:]\s*\d+)?\s*\|?", " ", s)
    s = re.sub(r"[\s|\[\]`'、。]+", " ", s).strip()
    return normalize_spaces(s), scale_a1, scale_a3


def normalize_name_for_compare(name: str) -> str:
    s = normalize_spaces(name)
    s = s.replace("·", " ").replace("_", " ").replace("-", " ")
    s = re.sub(r"\([^)]*\)", "", s)
    return normalize_spaces(s)
