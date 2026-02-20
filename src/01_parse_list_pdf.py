"""
도면목록표 PDF 파싱 - 바운딩박스 기반 (Tesseract)
PDF -> 이미지 -> 표 ROI -> OCR 바운딩박스 -> xy 축 기준 칼럼 분류
"""
import os
import json
import fitz
import cv2
import numpy as np
from PIL import Image
from utils_io import ensure_dirs, write_csv
from utils_ocr import ocr_to_boxes, is_gray_shaded_row, extract_rows_by_table_lines
from utils_parse import extract_list_no, normalize_spaces, normalize_scale, clean_scale_from_name

# 프로젝트 루트: src 한 단계 위. 중첩 구조면 drawing_checker 안쪽 사용
_script_dir = os.path.dirname(os.path.abspath(__file__))
_base = os.path.dirname(_script_dir)
_inner = os.path.join(_base, "drawing_checker")
if os.path.isdir(os.path.join(_base, "input")):
    BASE_DIR = _base  # 플랫: .../drawing_checker/input
else:
    BASE_DIR = _inner if os.path.isdir(_inner) else _base  # 중첩: .../drawing_checker/drawing_checker/input

INPUT_PDF = os.path.join(BASE_DIR, "input", "list_pdf", "도면목록표.pdf")
CFG_PATH = os.path.join(BASE_DIR, "input", "config", "roi_config.json")
OUT_DIR = os.path.join(BASE_DIR, "output", "list_index")
DBG_DIR = os.path.join(BASE_DIR, "debug", "list_rows")

SAVE_DEBUG = True
DEBUG_PAGES = {1, 2}
PAGE_LIMIT = None  # None=전체, 숫자=해당 페이지만 (테스트용)

# x 비율로 칼럼 구간 (패널 너비 대비 0~1, config에서 override 가능)
DEFAULT_COL_X_RANGES = {
    "drawing_no": (0, 0.25),
    "drawing_name": (0.25, 0.75),
    "scale_a1": (0.75, 0.88),
    "scale_a3": (0.88, 1.0),
}

ROW_Y_TOLERANCE = 25  # 같은 행: center_y 차이 이내 (px, 전처리 이미지 기준)
SCALE_PREPROCESS_SCALE = 2  # ocr_to_boxes 내부 2배 확대


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        msg = " ".join(str(a) for a in args)
        print(msg.encode("ascii", "replace").decode(), **kwargs)


def get_column_for_box(box: dict, panel_width: float, col_ranges: dict) -> str:
    """
    박스가 어느 칼럼에 속하는지 → 가로 겹침이 큰 칼럼 반환.
    (경계에서 잘리는 것 방지)
    """
    left_px = box["left"]
    right_px = box["left"] + box["width"]
    pw = panel_width
    if pw <= 0:
        return "drawing_name"

    best_col, best_overlap = "drawing_name", 0
    for col_name, (lo, hi) in col_ranges.items():
        col_left = lo * pw
        col_right = hi * pw
        overlap = max(0, min(right_px, col_right) - max(left_px, col_left))
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col_name
    return best_col


def get_row_bounds(gray_roi: np.ndarray, rows_cfg: dict) -> list[tuple[int, int]]:
    """표 행 경계 (균등 분할 또는 가로선 감지)"""
    rows = extract_rows_by_table_lines(gray_roi)
    if len(rows) >= 20:
        return rows
    h, _ = gray_roi.shape[:2]
    y0 = int(h * float(rows_cfg["y_start_ratio"]))
    y1 = int(h * float(rows_cfg["y_end_ratio"]))
    n_rows = int(rows_cfg["n_rows"])
    step = (y1 - y0) / n_rows
    out = []
    for i in range(n_rows):
        a = int(y0 + i * step)
        b = int(y0 + (i + 1) * step)
        if b - a >= 8:
            out.append((a, b))
    return out


def parse_panel_bbox(
    panel_gray: np.ndarray,
    row_bounds: list[tuple[int, int]],
    pi: int,
    pidx: int,
    gray_th: float,
    col_ranges: dict,
    save_debug: bool,
) -> list[dict]:
    """
    패널 OCR -> 바운딩박스 -> 행별(가장 가까운 행) 할당 -> 칼럼별(박스 겹침 기준) 분류
    경계 텍스트 잘림 방지
    """
    ph, pw = panel_gray.shape[:2]
    boxes = list(ocr_to_boxes(panel_gray, lang="kor+eng", min_conf=0))
    pw_scaled = pw * SCALE_PREPROCESS_SCALE

    # 각 박스를 center_y가 속한 행에 할당 (행 수 유지, 얇은 행도 유지)
    rows: dict[int, list] = {i: [] for i in range(len(row_bounds))}
    for b in boxes:
        cy = b["top"] + b["height"] / 2
        assigned = False
        for i, (y0, y1) in enumerate(row_bounds):
            row_top = y0 * SCALE_PREPROCESS_SCALE
            row_bottom = y1 * SCALE_PREPROCESS_SCALE
            if row_top <= cy <= row_bottom:
                rows[i].append(b)
                assigned = True
                break
        if not assigned:
            # 경계 근처: 가장 가까운 행에 할당
            best_i, best_dist = 0, float("inf")
            for i, (y0, y1) in enumerate(row_bounds):
                row_cy = (y0 + y1) / 2 * SCALE_PREPROCESS_SCALE
                if abs(cy - row_cy) < best_dist:
                    best_dist = abs(cy - row_cy)
                    best_i = i
            rows[best_i].append(b)

    out = []
    for i, (y0_orig, y1_orig) in enumerate(row_bounds):
        row_slice = panel_gray[y0_orig:y1_orig, :]
        if is_gray_shaded_row(row_slice, gray_th):
            continue

        row_boxes = rows[i]
        if not row_boxes:
            continue

        cells = {"drawing_no": [], "drawing_name": [], "scale_a1": [], "scale_a3": []}
        for b in row_boxes:
            col = get_column_for_box(b, pw_scaled, col_ranges)
            if col in cells:
                cells[col].append(b["text"])

        no_candidates = " ".join(cells["drawing_no"])
        drawing_no = extract_list_no(no_candidates)
        if not drawing_no:
            name_text = " ".join(cells["drawing_name"])
            drawing_no = extract_list_no(no_candidates + " " + name_text)

        if not drawing_no:
            continue

        drawing_name = normalize_spaces(" ".join(cells["drawing_name"]))

        a1_text = " ".join(cells["scale_a1"])
        a3_text = " ".join(cells["scale_a3"])
        scale_a1 = normalize_scale(a1_text)
        scale_a3 = normalize_scale(a3_text)
        if "NONE" in a1_text.upper():
            scale_a1 = scale_a1 or "NONE"
        if "NONE" in a3_text.upper():
            scale_a3 = scale_a3 or "NONE"

        drawing_name, s1, s2 = clean_scale_from_name(drawing_name)
        if not scale_a1 and s1:
            scale_a1 = s1
        if not scale_a3 and s2:
            scale_a3 = s2

        out.append({
            "list_page": pi,
            "panel": pidx,
            "drawing_no": drawing_no,
            "drawing_name": drawing_name,
            "scale_a1": scale_a1,
            "scale_a3": scale_a3,
        })

    return out


def main():
    if not os.path.exists(INPUT_PDF):
        safe_print(f"[X] 목록표 PDF 없음: {INPUT_PDF}")
        return

    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    ensure_dirs(OUT_DIR, DBG_DIR)
    safe_print("DBG_DIR =", DBG_DIR)

    roi_cfg = cfg["list_table_roi"]
    panels = cfg["list_panels"]
    rows_cfg = cfg["list_rows"]
    gray_th = float(cfg.get("list_gray_row_mid_ratio_threshold", 0.35))

    # list_columns → x 구간 변환 (비율 [a,b] -> (a, b))
    cols_cfg = cfg.get("list_columns", {})
    col_ranges = {}
    for k in ["drawing_no", "drawing_name", "scale_a1", "scale_a3"]:
        if k in cols_cfg and len(cols_cfg[k]) >= 2:
            col_ranges[k] = (float(cols_cfg[k][0]), float(cols_cfg[k][1]))
        elif k in DEFAULT_COL_X_RANGES:
            col_ranges[k] = DEFAULT_COL_X_RANGES[k]

    doc = fitz.open(INPUT_PDF)
    results = []
    n_pages = min(len(doc), PAGE_LIMIT) if PAGE_LIMIT else len(doc)

    for pi in range(1, n_pages + 1):
        page = doc[pi - 1]
        pix = page.get_pixmap(dpi=600)  # 450→600 해상도 향상 (텍스트 잘림 완화)
        pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        arr = np.array(pil)
        H, W = arr.shape[:2]

        x0 = int(W * roi_cfg["x0"])
        y0 = int(H * roi_cfg["y0"])
        x1 = int(W * roi_cfg["x1"])
        y1 = int(H * roi_cfg["y1"])
        roi = arr[y0:y1, x0:x1]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        save_debug = SAVE_DEBUG and (pi in DEBUG_PAGES)
        if save_debug:
            cv2.imwrite(os.path.join(DBG_DIR, f"p{pi:02d}_ROI.png"), gray_roi)

        rh, rw = gray_roi.shape[:2]
        row_bounds = get_row_bounds(gray_roi, rows_cfg)
        page_rows = []

        for pidx, (px0r, px1r) in enumerate(panels, start=1):
            px0 = int(rw * px0r)
            px1 = int(rw * px1r)
            panel = gray_roi[:, px0:px1]

            if save_debug:
                cv2.imwrite(os.path.join(DBG_DIR, f"p{pi:02d}_panel{pidx}.png"), panel)

            page_rows.extend(
                parse_panel_bbox(panel, row_bounds, pi, pidx, gray_th, col_ranges, save_debug)
            )

        results.extend(page_rows)
        safe_print(f"[OK] page {pi}/{n_pages} parsed rows: {len(page_rows)}")

    doc.close()

    out_csv = os.path.join(OUT_DIR, "list_index.csv")
    write_csv(
        out_csv, results,
        ["list_page", "panel", "drawing_no", "drawing_name", "scale_a1", "scale_a3"]
    )
    safe_print(f"\n[OK] 완료! 저장: {out_csv}")
    safe_print(f"디버그 폴더: {DBG_DIR}")


if __name__ == "__main__":
    main()
