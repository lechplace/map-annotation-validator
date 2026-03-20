"""
color_detector.py
-----------------
Klasyczna CV detekcja drzew (zielone okręgi) i dróg (żółte/szare pasy) w HSV.
Oblicza IoU między okręgiem drzewa a maską drogi.
"""

import cv2
import numpy as np


# ── Zakresy HSV ──────────────────────────────────────────────────────────────

# Zielony okrąg (oznaczenie drzewa)
GREEN_LOWER = np.array([35, 60, 40])
GREEN_UPPER = np.array([90, 255, 255])

# Droga żółta/pomarańczowa
YELLOW_LOWER = np.array([15, 80, 120])
YELLOW_UPPER = np.array([40, 255, 255])

# Droga szara (niska saturacja)
GRAY_SAT_MAX = 40
GRAY_VAL_MIN = 80
GRAY_VAL_MAX = 190


def to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


def mask_green(img_bgr: np.ndarray) -> np.ndarray:
    """Maska pikseli należących do zielonego okręgu drzewa."""
    hsv = to_hsv(img_bgr)
    return cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)


def mask_road(img_bgr: np.ndarray) -> np.ndarray:
    """Maska pikseli drogi (żółtej LUB szarej)."""
    hsv = to_hsv(img_bgr)
    yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    # Szara: niska saturacja, srednia jasność
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    gray = ((s < GRAY_SAT_MAX) & (v >= GRAY_VAL_MIN) & (v <= GRAY_VAL_MAX)).astype(np.uint8) * 255

    return cv2.bitwise_or(yellow, gray)


def detect_tree_circles(mask_green: np.ndarray, min_r: int = 5, max_r: int = 40):
    """
    Zwraca listę wykrytych okręgów drzew [(cx, cy, r), ...].
    Używa konturowania jako solidniejszej alternatywy dla HoughCircles
    przy małych patchach.
    """
    # Zamknij drobne luki w pierścieniu
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        r = int(r)
        if min_r <= r <= max_r:
            circles.append((int(cx), int(cy), r))
    return circles


def circle_mask(shape: tuple, cx: int, cy: int, r: int) -> np.ndarray:
    """Maska binarna wypełnionego koła o zadanym centrum i promieniu."""
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU między dwoma maskami binarnymi."""
    inter = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return inter / union if union > 0 else 0.0


def classify_patch(img_bgr: np.ndarray, iou_threshold: float = 0.05) -> int:
    """
    Klasyfikuje patch jako NOT-OK (1) lub OK (0).

    NOT-OK: jakikolwiek wykryty okrąg drzewa nakłada się na drogę z IoU > próg.
    OK:     brak zielonego okręgu LUB brak nakładania.

    Returns
    -------
    int  0 = OK, 1 = NOT-OK, -1 = brak drzewa (pomiń)
    """
    g_mask = mask_green(img_bgr)
    r_mask = mask_road(img_bgr)

    circles = detect_tree_circles(g_mask)
    if not circles:
        return -1  # brak drzewa → pomiń

    for (cx, cy, r) in circles:
        c_mask = circle_mask(img_bgr.shape, cx, cy, r)
        iou = compute_iou(c_mask, r_mask)
        if iou > iou_threshold:
            return 1  # NOT-OK

    return 0  # OK


def visualize_patch(img_bgr: np.ndarray, iou_threshold: float = 0.05) -> np.ndarray:
    """Zwraca obraz z nałożonymi maskami i wykrytymi okręgami (do debugowania)."""
    vis = img_bgr.copy()
    g_mask = mask_green(img_bgr)
    r_mask = mask_road(img_bgr)

    # Zielona nakładka = maska drzewa
    vis[g_mask > 0] = (0, 200, 0)
    # Czerwona nakładka = maska drogi
    vis[r_mask > 0] = (0, 0, 200)

    circles = detect_tree_circles(g_mask)
    for (cx, cy, r) in circles:
        c_mask = circle_mask(img_bgr.shape, cx, cy, r)
        iou = compute_iou(c_mask, r_mask)
        color = (0, 0, 255) if iou > iou_threshold else (0, 255, 0)
        cv2.circle(vis, (cx, cy), r, color, 1)

    return vis
