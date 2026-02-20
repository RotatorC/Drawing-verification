import cv2
import numpy as np
import pytesseract
from typing import Generator

def _preprocess_for_ocr(img: np.ndarray, pad: int = 4) -> np.ndarray:
    """OCR 전처리: 패딩(잘림 방지) -> 2배 확대 -> 블러 -> 적응형 이진화"""
    if img.size == 0:
        return img
    # 테두리 패딩 (경계 글자 잘림 방지)
    if pad > 0:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        31, 8  # C=8 (11→8) 이진화 완화
    )
    return img


def ocr_line(gray: np.ndarray, lang: str, config: str) -> str:
    img = _preprocess_for_ocr(gray)
    return pytesseract.image_to_string(img, lang=lang, config=config).strip()


def ocr_to_boxes(gray: np.ndarray, lang: str = "kor+eng", min_conf: int = 0) -> Generator[dict, None, None]:
    """
    OCR 바운딩 박스 반환.
    yield: {"text": str, "left": int, "top": int, "width": int, "height": int}
    """
    img = _preprocess_for_ocr(gray)
    # PSM 6: 블록 단위 (표 셀에 적합), 패딩 적용되므로 left/top 보정 필요
    pad = 4
    config = r"--psm 6 --oem 3"
    data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        conf = int(data["conf"][i] or 0)
        if conf < min_conf:
            continue
        # 패딩 보정 (좌표를 원본 기준으로)
        left = data["left"][i] - pad * 2  # 2x 확대 후 패딩 픽셀
        top = data["top"][i] - pad * 2
        if left < 0:
            left = 0
        if top < 0:
            top = 0
        yield {
            "text": text,
            "left": left,
            "top": top,
            "width": data["width"][i],
            "height": data["height"][i],
        }

def is_gray_shaded_row(gray_row: np.ndarray, mid_ratio_threshold: float = 0.35) -> bool:
    mid = ((gray_row >= 90) & (gray_row <= 200)).mean()
    return mid > mid_ratio_threshold

def extract_rows_by_table_lines(gray_roi: np.ndarray) -> list[tuple[int,int]]:
    inv = 255 - gray_roi
    bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    hlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    ys = np.where(hlines.sum(axis=1) > 0)[0]
    if len(ys) < 10:
        return []

    lines = []
    start = ys[0]
    prev = ys[0]
    for y in ys[1:]:
        if y == prev + 1:
            prev = y
        else:
            lines.append((start + prev) // 2)
            start = prev = y
    lines.append((start + prev) // 2)

    rows = []
    for y0, y1 in zip(lines[:-1], lines[1:]):
        if (y1 - y0) < 12:
            continue
        rows.append((y0, y1))
    return rows
