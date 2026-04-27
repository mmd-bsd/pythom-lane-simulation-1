import cv2
import numpy as np
import os
import time
import json

# =====================================================================
# Lane v8 - Same v7 detection, with an interactive CALIBRATION phase.
#
# Workflow at startup:
#   1. Pick / load ROI                                  (roi_v8.npy)
#   2. Calibration loop:
#        - "Controls" window with sliders for every threshold/kernel
#        - 8 diagnostic windows show every pipeline stage
#        - SPACE pause, n next frame, p prev frame (video)
#        - s save preset, r reload preset, g GO (run main loop), ESC quit
#        - Preset auto-saved to tune_v8.json on GO
#   3. Main detection loop (press 'c' to re-enter calibration)
#
# The DETECTION algorithm is identical to lane_v7.py - only thresholds
# and kernel sizes are now parameterized so they can be slid live.
# =====================================================================

ROI_FILE  = "roi_v8.npy"
TUNE_FILE = "tune_v8.json"

TARGET_WIDTH  = 480
TARGET_HEIGHT = 270

# ---- DEFAULT PARAMS (mirror lane_v7.py) ----
DEFAULT_PARAMS = {
    "T_BLACKHAT":     22,
    "HSV_V_MAX":      98,
    "HSV_S_MAX":      170,
    "LAB_L_MAX":     105,
    "BH_K_W":         15,   # black-hat kernel width  (odd)
    "BH_K_H":          5,   # black-hat kernel height (odd)
    "OPEN_K":          3,   # open kernel side (odd)
    "CLOSE_K_W":       3,   # close kernel width  (odd)
    "CLOSE_K_H":      15,   # close kernel height (odd)
    "MIN_AREA":       60,
    "MIN_H_FRAC":   0.10,
    "MIN_ASPECT":   1.40,
    "MAX_RMS":      12.0,
    "MAX_CURV":   0.0020,    # max |a| in polyfit x = a y^2 + b y + c
    "Y_REF_FRAC":   0.85,
    "LOOKAHEAD_FRAC": 0.65,
    "DEAD_BAND":      8,
    "TRACK_GATE":    70,
    "LOSE_FRAMES":    8,
    "EMA_ALPHA":   0.20,
    "LANE_EMA":    0.30,    # per-side polyfit EMA
    "HOLD_FRAMES":   20,    # how many frames a missing line stays "virtual"
    "KP":          0.15,
    "KD":          0.40,
}

CONTROLS_WIN = "Controls"

# ---- ROI selection state ----
clicked_points = []
clone_img      = None


# ---------- TRACK STATE ----------
class TrackState:
    def __init__(self):
        # Per-side smoothed polyfit + smoothed reference x.
        # `health` rises on detection, decays on miss; a side is shown
        # (real or virtual) until its health falls below HEALTH_KEEP_MIN.
        self.left_fit       = None      # smoothed (3,) polyfit
        self.right_fit      = None
        self.left_x_ref     = None
        self.right_x_ref    = None
        self.left_y_min     = None      # smoothed observed y range,
        self.left_y_max     = None      # used to bound the drawn line
        self.right_y_min    = None
        self.right_y_max    = None
        self.left_health    = 0.0       # [0, 1]
        self.right_health   = 0.0
        self.half_width_px  = TARGET_WIDTH * 0.25
        self.smoothed_center_fit = None

state = TrackState()


def reset_state():
    global state
    state = TrackState()


def resize_image(img):
    return cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))


def _odd(v):
    v = max(1, int(v))
    return v if v % 2 else v + 1


# =====================================================================
# ROI
# =====================================================================
def mouse_callback(event, x, y, flags, param):
    global clicked_points, clone_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            cv2.circle(clone_img, (x, y), 5, (0, 255, 0), -1)
            if len(clicked_points) > 1:
                cv2.line(clone_img, tuple(clicked_points[-2]), tuple(clicked_points[-1]), (0, 255, 0), 2)
            if len(clicked_points) == 4:
                cv2.line(clone_img, tuple(clicked_points[-1]), tuple(clicked_points[0]), (0, 255, 0), 2)
            cv2.imshow("Select ROI", clone_img)


def select_and_save_roi(img):
    global clicked_points, clone_img
    clicked_points = []
    clone_img = img.copy()

    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("Select ROI", clone_img)
    print("Click 4 points for ROI (TL -> TR -> BR -> BL). ESC to cancel.")
    cv2.setMouseCallback("Select ROI", mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(clicked_points) == 4:
            cv2.waitKey(400)
            break
        if key == 27:
            break

    cv2.destroyWindow("Select ROI")

    if len(clicked_points) == 4:
        pts = np.array(clicked_points, dtype=np.float32)
        h, w = img.shape[:2]
        pts[:, 0] /= w
        pts[:, 1] /= h
        np.save(ROI_FILE, pts)
        print("ROI saved.")
        return pts
    return None


def load_roi(img):
    if os.path.exists(ROI_FILE):
        print("Loaded existing ROI.")
        return np.load(ROI_FILE)
    print("No ROI yet - please pick one.")
    return select_and_save_roi(img)


# =====================================================================
# PRESET PERSISTENCE
# =====================================================================
def load_params():
    p = dict(DEFAULT_PARAMS)
    if os.path.exists(TUNE_FILE):
        try:
            with open(TUNE_FILE, "r") as f:
                user = json.load(f)
            for k in p:
                if k in user:
                    p[k] = type(p[k])(user[k])
            print(f"Loaded tuning preset from {TUNE_FILE}.")
        except Exception as e:
            print(f"Could not read {TUNE_FILE} ({e}); using defaults.")
    else:
        print("No tuning preset; using v7 defaults.")
    return p


def save_params(params):
    try:
        with open(TUNE_FILE, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved preset to {TUNE_FILE}.")
    except Exception as e:
        print(f"Could not save preset: {e}")


# =====================================================================
# AUTOTUNE
#
# One pass over the source: sample N frames, compute statistics, then
# pick parameters from those statistics. The result is a permissive
# starting point that ALREADY detects the lines; the user can tighten
# manually from there.
#
# Strategy per parameter:
#   LAB_L_MAX  - histogram mode of the L channel = floor brightness;
#                set ceiling 25 below it.
#   BH_K_W     - try {9, 13, 17, 21}; pick width with biggest
#                blackhat p99 - p50 (signal vs background).
#   T_BLACKHAT - Otsu threshold on the chosen blackhat values, then
#                relax by 30% so faint lines still pass.
#   HSV gates  - sample HSV at pixels that pass the chosen blackhat
#                threshold; take the 98th percentile of S and V on
#                line pixels and pad slightly.
#   MIN_AREA / MIN_H_FRAC / MIN_ASPECT - run a candidate-finding pass
#                with the gates above, keep the bottom 20th-percentile
#                stats of "tall vertical" contours, scale by 0.6.
# =====================================================================
def _sample_frames(cap_or_img, source_kind, n_samples):
    if source_kind == "image":
        return [cap_or_img]
    n_total = int(cap_or_img.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if source_kind == "video" and n_total > 0:
        idxs = np.linspace(0, max(0, n_total - 1), n_samples).astype(int)
        for i in idxs:
            cap_or_img.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, fr = cap_or_img.read()
            if ret:
                frames.append(fr)
        cap_or_img.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        # camera: grab n_samples frames spaced ~50 ms apart
        for _ in range(n_samples):
            ret, fr = cap_or_img.read()
            if ret:
                frames.append(fr)
            time.sleep(0.05)
    return frames


def autotune_params(cap_or_img, roi_norm, source_kind, n_samples=30, verbose=True):
    if verbose:
        print("\n=== AUTOTUNE: sampling frames... ===")
    frames = _sample_frames(cap_or_img, source_kind, n_samples)
    if not frames:
        print("Autotune: no frames available, keeping defaults.")
        return dict(DEFAULT_PARAMS)
    if verbose:
        print(f"Autotune: collected {len(frames)} frames.")

    H, W = TARGET_HEIGHT, TARGET_WIDTH
    src = roi_norm.copy().astype(np.float32)
    src[:, 0] *= W; src[:, 1] *= H
    horizon_y = int(min(src[0][1], src[1][1]))

    # ----- Phase 1: L-channel floor -----
    L_roads = []
    for fr in frames:
        img = resize_image(fr)
        L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
        L_roads.append(L[horizon_y:H, :])
    L_all = np.concatenate([l.ravel() for l in L_roads])
    hist, edges = np.histogram(L_all, bins=64, range=(0, 256))
    floor_mode = int(edges[hist.argmax()] + (edges[1] - edges[0]) / 2)
    lab_l_max = int(np.clip(floor_mode - 25, 70, 170))
    if verbose:
        print(f"Autotune: floor L mode ~ {floor_mode} -> LAB_L_MAX = {lab_l_max}")

    # ----- Phase 2: black-hat kernel width search -----
    candidate_widths = [9, 13, 17, 21]
    best_w, best_score = 15, -1.0
    L_blurs = [cv2.GaussianBlur(l, (5, 5), 0) for l in L_roads]
    for kw in candidate_widths:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 5))
        scores = []
        # subsample frames for speed
        for L_blur in L_blurs[::3] or [L_blurs[0]]:
            bh = cv2.morphologyEx(L_blur, cv2.MORPH_BLACKHAT, kernel)
            p99 = float(np.percentile(bh, 99))
            p50 = float(np.percentile(bh, 50))
            scores.append(p99 - p50)
        s = float(np.median(scores))
        if s > best_score:
            best_w, best_score = kw, s
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (best_w, 5))
    if verbose:
        print(f"Autotune: black-hat kernel width = {best_w} (signal {best_score:.1f})")

    # ----- Phase 3: T_BLACKHAT via Otsu on non-trivial bh values -----
    bh_pool = []
    for L_blur in L_blurs:
        bh = cv2.morphologyEx(L_blur, cv2.MORPH_BLACKHAT, bh_kernel)
        bh_pool.append(bh)
    bh_concat = np.concatenate([b.ravel() for b in bh_pool])
    nonzero = bh_concat[bh_concat > 3]
    if nonzero.size > 1000:
        otsu_t, _ = cv2.threshold(
            nonzero.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        T_blackhat = int(np.clip(otsu_t * 0.7, 6, 60))
    else:
        T_blackhat = 12
    if verbose:
        print(f"Autotune: T_BLACKHAT = {T_blackhat}")

    # ----- Phase 4: HSV gates from blackhat-positive pixels -----
    Vs, Ss = [], []
    for fr, bh in zip(frames, bh_pool):
        img = resize_image(fr)
        line_mask = (bh > T_blackhat).astype(np.uint8)
        if line_mask.sum() < 50:
            continue
        hsv_road = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[horizon_y:H, :]
        Vs.append(hsv_road[:, :, 2][line_mask.astype(bool)])
        Ss.append(hsv_road[:, :, 1][line_mask.astype(bool)])
    if Vs:
        V_all = np.concatenate(Vs); S_all = np.concatenate(Ss)
        hsv_v_max = int(min(180, np.percentile(V_all, 98) + 15))
        hsv_s_max = int(min(220, np.percentile(S_all, 98) + 20))
    else:
        hsv_v_max, hsv_s_max = 130, 110
    # Floor: never tighter than default v7 numbers
    hsv_v_max = max(hsv_v_max, 95)
    hsv_s_max = max(hsv_s_max, 90)
    if verbose:
        print(f"Autotune: HSV_V_MAX = {hsv_v_max}, HSV_S_MAX = {hsv_s_max}")

    # ----- Phase 5: Shape filter floors from real contour stats -----
    # Build a permissive partial pipeline using the params chosen above.
    p_partial = dict(DEFAULT_PARAMS)
    p_partial.update(dict(
        T_BLACKHAT=T_blackhat,
        HSV_V_MAX=hsv_v_max, HSV_S_MAX=hsv_s_max, LAB_L_MAX=lab_l_max,
        BH_K_W=best_w,
        # very loose shape gates so we observe whatever contours are there
        MIN_AREA=15, MIN_H_FRAC=0.04, MIN_ASPECT=0.8, MAX_RMS=40,
    ))

    dst = np.float32([
        [W * 0.25, 0], [W * 0.75, 0],
        [W * 0.75, H], [W * 0.25, H],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))

    areas, h_fracs, aspects = [], [], []
    for fr in frames[::2] or [frames[0]]:
        masked, _stages = build_black_mask(resize_image(fr), horizon_y, p_partial)
        warped = cv2.warpPerspective(masked, M, (W, H))
        cleaned = cv2.morphologyEx(warped, cv2.MORPH_OPEN, open_k)
        closed_ = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_k)
        contours, _ = cv2.findContours(closed_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0:
                continue
            aspect = float(h) / float(w)
            if aspect < 0.8:    # only keep "vertical-ish" things
                continue
            a = cv2.contourArea(cnt)
            if a < 15:
                continue
            areas.append(a)
            h_fracs.append(h / float(H))
            aspects.append(aspect)

    if areas:
        min_area   = int(max(20, np.percentile(areas, 20) * 0.6))
        min_h_frac = float(max(0.04, np.percentile(h_fracs, 20) * 0.6))
        min_aspect = float(max(0.9, np.percentile(aspects, 20) * 0.6))
    else:
        # Nothing detected: fall back to extra-permissive
        min_area, min_h_frac, min_aspect = 25, 0.05, 0.9
    if verbose:
        print(f"Autotune: MIN_AREA={min_area} MIN_H_FRAC={min_h_frac:.3f} MIN_ASPECT={min_aspect:.2f}")

    # ----- Compose final params -----
    p = dict(DEFAULT_PARAMS)
    p["LAB_L_MAX"]   = lab_l_max
    p["BH_K_W"]      = best_w
    p["BH_K_H"]      = 5
    p["T_BLACKHAT"]  = T_blackhat
    p["HSV_V_MAX"]   = hsv_v_max
    p["HSV_S_MAX"]   = hsv_s_max
    p["OPEN_K"]      = 3
    p["CLOSE_K_W"]   = 3
    p["CLOSE_K_H"]   = 15
    p["MIN_AREA"]    = min_area
    p["MIN_H_FRAC"]  = round(min_h_frac, 3)
    p["MIN_ASPECT"]  = round(min_aspect, 2)
    p["MAX_RMS"]     = 14.0   # slightly looser than v7

    if verbose:
        print("Autotune: done.\n")
    return p


# =====================================================================
# CORE PIPELINE STAGES (parameterized)
# =====================================================================
def build_black_mask(img, horizon_y, p):
    height, width = img.shape[:2]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    L_road = L[horizon_y:height, :]
    L_blur = cv2.GaussianBlur(L_road, (5, 5), 0)

    bh_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (_odd(p["BH_K_W"]), _odd(p["BH_K_H"])))
    blackhat = cv2.morphologyEx(L_blur, cv2.MORPH_BLACKHAT, bh_kernel)
    _, bh_mask = cv2.threshold(blackhat, int(p["T_BLACKHAT"]), 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_road = hsv[horizon_y:height, :]
    lower = np.array([0, 0, 0], dtype=np.uint8)
    upper = np.array([180, int(p["HSV_S_MAX"]), int(p["HSV_V_MAX"])], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv_road, lower, upper)

    _, lab_mask = cv2.threshold(L_road, int(p["LAB_L_MAX"]), 255, cv2.THRESH_BINARY_INV)

    combined = cv2.bitwise_and(bh_mask, hsv_mask)
    combined = cv2.bitwise_and(combined, lab_mask)
    combined = cv2.medianBlur(combined, 3)

    full = np.zeros((height, width), dtype=np.uint8)
    full[horizon_y:height, :] = combined

    # Also build full-frame versions of every per-stage mask, for diagnostics
    def _full(mask_road):
        f = np.zeros((height, width), dtype=np.uint8)
        f[horizon_y:height, :] = mask_road
        return f

    stages = {
        "L_road":   _full(L_road),
        "blackhat": _full(blackhat),
        "bh_mask":  _full(bh_mask),
        "hsv_mask": _full(hsv_mask),
        "lab_mask": _full(lab_mask),
        "combined": full,
    }
    return full, stages


REJECT_REASONS = (
    "area",      # below MIN_AREA
    "height",    # below height * MIN_H_FRAC
    "aspect",    # below MIN_ASPECT
    "thin",      # too few boundary points
    "fit",       # polyfit linalg failure
    "curve",     # |a| exceeds MAX_CURV
    "rms",       # residual exceeds MAX_RMS
    "side",      # x_ref outside [5%, 95%] of width
)


def fit_candidates(closed, height, width, p, return_rejects=False):
    """Return list of accepted candidates. If return_rejects=True, also
    return a list of (contour, reason_string) for everything that was
    discarded by a shape gate -- this is what feeds the diagnostic
    window so the user can see which gate is killing a clear line."""
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cands   = []
    rejects = [] if return_rejects else None
    y_ref   = height * float(p["Y_REF_FRAC"])
    max_curv = float(p["MAX_CURV"])

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < float(p["MIN_AREA"]):
            if return_rejects: rejects.append((cnt, "area"))
            continue
        if h < height * float(p["MIN_H_FRAC"]):
            if return_rejects: rejects.append((cnt, "height"))
            continue
        if w == 0:
            if return_rejects: rejects.append((cnt, "aspect"))
            continue
        aspect = float(h) / float(w)
        if aspect < float(p["MIN_ASPECT"]):
            if return_rejects: rejects.append((cnt, "aspect"))
            continue

        pts = cnt.reshape(-1, 2)
        xs = pts[:, 0].astype(np.float32)
        ys = pts[:, 1].astype(np.float32)
        if len(xs) < 25:
            if return_rejects: rejects.append((cnt, "thin"))
            continue

        try:
            fit = np.polyfit(ys, xs, 2)
        except np.linalg.LinAlgError:
            if return_rejects: rejects.append((cnt, "fit"))
            continue

        if abs(fit[0]) > max_curv:
            if return_rejects: rejects.append((cnt, "curve"))
            continue

        pred = fit[0] * ys * ys + fit[1] * ys + fit[2]
        rms = float(np.sqrt(np.mean((pred - xs) ** 2)))
        if rms > float(p["MAX_RMS"]):
            if return_rejects: rejects.append((cnt, "rms"))
            continue

        x_ref = float(fit[0] * y_ref * y_ref + fit[1] * y_ref + fit[2])
        if x_ref < width * 0.05 or x_ref > width * 0.95:
            if return_rejects: rejects.append((cnt, "side"))
            continue

        cands.append({
            "fit": fit, "x_ref": x_ref,
            "y_min": float(np.min(ys)), "y_max": float(np.max(ys)),
            "rms": rms, "contour": cnt,
        })
    if return_rejects:
        return cands, rejects
    return cands


def pick_nearest(pool, car_x, prev_x, p):
    """Return the candidate dict closest to car_x, gated against prev_x.
    Returns None if nothing qualifies."""
    if not pool:
        return None
    if prev_x is not None:
        gated = [c for c in pool if abs(c["x_ref"] - prev_x) <= float(p["TRACK_GATE"])]
        if gated:
            pool = gated
    return min(pool, key=lambda c: abs(c["x_ref"] - car_x))


HEALTH_GAIN     = 0.25     # rise per detection
HEALTH_KEEP_MIN = 0.05     # below this, the side is fully dropped


def _update_side(side, picked, p):
    """Update state.<side>_fit / x_ref / y_min / y_max / health from a fresh
    pick. side is 'left' or 'right'."""
    cur_fit  = getattr(state, side + "_fit")
    cur_x    = getattr(state, side + "_x_ref")
    cur_ymin = getattr(state, side + "_y_min")
    cur_ymax = getattr(state, side + "_y_max")
    cur_h    = getattr(state, side + "_health")
    lane_a   = float(p["LANE_EMA"])
    decay    = 1.0 / max(1.0, float(p["HOLD_FRAMES"]))

    if picked is not None:
        new_fit  = picked["fit"]
        new_x    = float(picked["x_ref"])
        new_ymin = float(picked["y_min"])
        new_ymax = float(picked["y_max"])
        if cur_fit is None:
            cur_fit, cur_x = new_fit, new_x
            cur_ymin, cur_ymax = new_ymin, new_ymax
        else:
            cur_fit  = lane_a * new_fit  + (1 - lane_a) * cur_fit
            cur_x    = lane_a * new_x    + (1 - lane_a) * cur_x
            cur_ymin = lane_a * new_ymin + (1 - lane_a) * cur_ymin
            cur_ymax = lane_a * new_ymax + (1 - lane_a) * cur_ymax
        cur_h = min(1.0, cur_h + HEALTH_GAIN)
    else:
        cur_h = max(0.0, cur_h - decay)

    setattr(state, side + "_fit",    cur_fit)
    setattr(state, side + "_x_ref",  cur_x)
    setattr(state, side + "_y_min",  cur_ymin)
    setattr(state, side + "_y_max",  cur_ymax)
    setattr(state, side + "_health", cur_h)


# =====================================================================
# PIPELINE (unified for main loop and calibration)
# =====================================================================
def pipeline(frame, roi_norm, p, debug=False):
    img = resize_image(frame)
    height, width = img.shape[:2]

    src = roi_norm.copy().astype(np.float32)
    src[:, 0] *= width
    src[:, 1] *= height
    horizon_y = int(min(src[0][1], src[1][1]))

    masked, stages = build_black_mask(img, horizon_y, p)

    dst = np.float32([
        [width * 0.25, 0], [width * 0.75, 0],
        [width * 0.75, height], [width * 0.25, height],
    ])
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(masked, M, (width, height))

    open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(p["OPEN_K"]),  _odd(p["OPEN_K"])))
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(p["CLOSE_K_W"]), _odd(p["CLOSE_K_H"])))
    cleaned = cv2.morphologyEx(warped, cv2.MORPH_OPEN, open_k)
    closed  = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_k)

    if debug:
        cands, rejects = fit_candidates(closed, height, width, p, return_rejects=True)
    else:
        cands = fit_candidates(closed, height, width, p)
        rejects = None

    car_x = width / 2.0
    db    = float(p["DEAD_BAND"])
    left_pool  = [c for c in cands if c["x_ref"] < car_x - db]
    right_pool = [c for c in cands if c["x_ref"] > car_x + db]

    # Pick best raw candidate per side, gated against the smoothed previous x
    left_pick  = pick_nearest(left_pool,  car_x, state.left_x_ref,  p)
    right_pick = pick_nearest(right_pool, car_x, state.right_x_ref, p)

    # Update smoothed per-side fits and health
    _update_side("left",  left_pick,  p)
    _update_side("right", right_pick, p)

    # Adaptive half lane width - update only on FRESH both-side detection
    if left_pick is not None and right_pick is not None:
        measured = (right_pick["x_ref"] - left_pick["x_ref"]) / 2.0
        if measured > 30:
            state.half_width_px = 0.8 * state.half_width_px + 0.2 * measured

    left_active  = state.left_fit  is not None and state.left_health  > HEALTH_KEEP_MIN
    right_active = state.right_fit is not None and state.right_health > HEALTH_KEEP_MIN

    # Rebuild mode label
    if left_active and right_active:
        if left_pick is not None and right_pick is not None:
            rebuild_mode = "2 Lanes"
        elif left_pick is None and right_pick is None:
            rebuild_mode = "2 Lanes (virtual)"
        else:
            rebuild_mode = "2 Lanes (1 virtual)"
    elif left_active:
        rebuild_mode = "1 Lane (Left)" if left_pick is not None else "1 Virtual (Left)"
    elif right_active:
        rebuild_mode = "1 Lane (Right)" if right_pick is not None else "1 Virtual (Right)"
    else:
        rebuild_mode = "No Lanes"

    # Center fit from smoothed sides - always present while >=1 side active
    center_fit = None
    if left_active and right_active:
        center_fit = (state.left_fit + state.right_fit) / 2.0
    elif left_active:
        center_fit = state.left_fit.copy(); center_fit[2] += state.half_width_px
    elif right_active:
        center_fit = state.right_fit.copy(); center_fit[2] -= state.half_width_px

    alpha = float(p["EMA_ALPHA"])
    if center_fit is not None:
        if state.smoothed_center_fit is None:
            state.smoothed_center_fit = center_fit
        else:
            state.smoothed_center_fit = alpha * center_fit + (1 - alpha) * state.smoothed_center_fit
    elif state.smoothed_center_fit is not None:
        rebuild_mode = "History Fallback"

    color_warp = np.zeros((height, width, 3), dtype=np.uint8)
    for c in cands:
        f = c["fit"]
        ploty = np.linspace(c["y_min"], c["y_max"], 80)
        fitx = f[0] * ploty * ploty + f[1] * ploty + f[2]
        pts = np.array([np.transpose(np.vstack([fitx, ploty]))], dtype=np.int32)
        cv2.polylines(color_warp, pts, False, (90, 90, 90), 4)

    # Draw each side ONLY within its observed y range (with a small
    # extension toward the bottom so the line reaches the car). Drawing
    # the whole 0..height-1 range on a short contour produced a wildly
    # extrapolated line - that was the bug1.png artifact.
    EXT_PAD = 25  # pixels of allowed extrapolation each end

    def _draw_side(fit, y_min, y_max, color):
        if fit is None or y_min is None or y_max is None:
            return
        y_lo = max(0.0, float(y_min) - EXT_PAD)
        y_hi = min(float(height - 1), float(y_max) + EXT_PAD)
        if y_hi - y_lo < 5:
            return
        ploty = np.linspace(y_lo, y_hi, max(20, int(y_hi - y_lo)))
        fitx = fit[0] * ploty * ploty + fit[1] * ploty + fit[2]
        pts = np.array([np.transpose(np.vstack([fitx, ploty]))], dtype=np.int32)
        cv2.polylines(color_warp, pts, False, color, 14)

    if left_active:
        intensity = int(np.clip(80 + 175 * state.left_health, 80, 255))
        _draw_side(state.left_fit, state.left_y_min, state.left_y_max,
                   (intensity, 0, 0))
    if right_active:
        intensity = int(np.clip(80 + 175 * state.right_health, 80, 255))
        _draw_side(state.right_fit, state.right_y_min, state.right_y_max,
                   (0, 0, intensity))

    steer_deg = 0.0
    if state.smoothed_center_fit is not None:
        f = state.smoothed_center_fit
        # Drawing range = union of whichever sides are active, padded.
        ymins, ymaxs = [], []
        if left_active and state.left_y_min is not None:
            ymins.append(state.left_y_min); ymaxs.append(state.left_y_max)
        if right_active and state.right_y_min is not None:
            ymins.append(state.right_y_min); ymaxs.append(state.right_y_max)
        if ymins:
            y_lo = max(0.0, float(min(ymins)) - 25)
            y_hi = min(float(height - 1), float(max(ymaxs)) + 25)
        else:
            y_lo, y_hi = 0.0, float(height - 1)
        ploty = np.linspace(y_lo, y_hi, max(20, int(y_hi - y_lo)))
        cx = f[0] * ploty * ploty + f[1] * ploty + f[2]
        pts_c = np.array([np.transpose(np.vstack([cx, ploty]))], dtype=np.int32)
        cv2.polylines(color_warp, pts_c, False, (0, 255, 0), 60)

        lookahead_y = height * float(p["LOOKAHEAD_FRAC"])
        target_x = f[0] * lookahead_y * lookahead_y + f[1] * lookahead_y + f[2]
        cte = target_x - car_x
        slope = 2 * f[0] * lookahead_y + f[1]
        heading_err = np.degrees(np.arctan(slope))
        steer_deg = float(np.clip(float(p["KP"]) * cte + float(p["KD"]) * heading_err, -30, 30))
        cv2.circle(color_warp, (int(target_x), int(lookahead_y)), 12, (0, 165, 255), -1)

    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))
    result  = cv2.addWeighted(img, 1, newwarp, 0.8, 0)

    if not debug:
        return result, closed, color_warp, steer_deg, rebuild_mode

    # Draw ROI poly on a copy of the original for the diagnostic window
    roi_view = img.copy()
    poly = src.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(roi_view, [poly], True, (0, 255, 255), 2)

    # ----- 7b: contour-rejection diagnostic ----------------------------
    # Shows every contour from `closed` so you can see WHY a clear line
    # is being killed by the shape gates.
    #   GREEN   = accepted (became a candidate)
    #   colored = rejected; color encodes the reason:
    #     red       MIN_AREA
    #     orange    MIN_H_FRAC (height too small)
    #     yellow    MIN_ASPECT (too wide / horizontal)
    #     magenta   |a| > MAX_CURV (over-curved fit, was the bug1 case)
    #     cyan      RMS > MAX_RMS (noisy outline)
    #     white     x_ref outside [5%, 95%] band
    #     gray      misc (thin contour, fit error)
    REASON_COLORS = {
        "area":    (0,   0,  220),  # red
        "height":  (0, 140,  255),  # orange
        "aspect":  (0, 220,  255),  # yellow
        "curve":   (220, 0,  220),  # magenta
        "rms":     (220,220,   0),  # cyan
        "side":    (255,255, 255),  # white
        "thin":    (140,140, 140),
        "fit":     (140,140, 140),
    }
    filter_view = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR) // 3   # dim the bg
    if rejects:
        for cnt, reason in rejects:
            col = REASON_COLORS.get(reason, (180, 180, 180))
            cv2.drawContours(filter_view, [cnt], -1, col, 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(filter_view, reason, (x, max(10, y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    for c in cands:
        cv2.drawContours(filter_view, [c["contour"]], -1, (0, 255, 0), 2)

    diag = {
        "1_original":  roi_view,
        "2_blackhat":  stages["blackhat"],
        "3_bh_mask":   stages["bh_mask"],
        "4_hsv_mask":  stages["hsv_mask"],
        "5_lab_mask":  stages["lab_mask"],
        "6_combined":  stages["combined"],
        "7_closed":    closed,
        "7b_filter":   filter_view,
        "8_overlay":   result,
        "9_warp_dbg":  color_warp,
    }
    return result, closed, color_warp, steer_deg, rebuild_mode, diag


# =====================================================================
# CALIBRATION UI
# =====================================================================
TRACKBARS = [
    # (label, param key, max, scale)  -- displayed value = int; real = int*scale
    ("T_blackhat",     "T_BLACKHAT",      80,  1.0),
    ("HSV V_max",      "HSV_V_MAX",      255,  1.0),
    ("HSV S_max",      "HSV_S_MAX",      255,  1.0),
    ("Lab L_max",      "LAB_L_MAX",      255,  1.0),
    ("BH kernel w",    "BH_K_W",          31,  1.0),
    ("BH kernel h",    "BH_K_H",          15,  1.0),
    ("Open k",         "OPEN_K",           9,  1.0),
    ("Close k w",      "CLOSE_K_W",       25,  1.0),
    ("Close k h",      "CLOSE_K_H",       41,  1.0),
    ("Min area",       "MIN_AREA",       500,  1.0),
    ("Min H pct",      "MIN_H_FRAC",      50,  0.01),
    ("Min aspect x10", "MIN_ASPECT",      60,  0.1),
    ("Max RMS",        "MAX_RMS",         40,  1.0),
    ("Max curve x10k", "MAX_CURV",        100,  0.0001),
    ("Y_ref pct",      "Y_REF_FRAC",      95,  0.01),
    ("Lookahead pct",  "LOOKAHEAD_FRAC",  90,  0.01),
    ("Dead band",      "DEAD_BAND",       30,  1.0),
    ("Track gate",     "TRACK_GATE",     150,  1.0),
    ("Lose frames",    "LOSE_FRAMES",     30,  1.0),
    ("EMA alpha x100", "EMA_ALPHA",       80,  0.01),
    ("Lane EMA x100",  "LANE_EMA",        90,  0.01),
    ("Hold frames",    "HOLD_FRAMES",     60,  1.0),
    ("Kp x100",        "KP",              60,  0.01),
    ("Kd x100",        "KD",             100,  0.01),
]

# Buttons (separate from numeric trackbars so they don't get saved as params)
BUTTON_PAUSE = "Pause (0/1)"


def _nop(_):
    pass


def _build_controls(params):
    cv2.namedWindow(CONTROLS_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WIN, 460, 760)
    # Pause "button": 0 = running, 1 = frozen on current frame
    cv2.createTrackbar(BUTTON_PAUSE, CONTROLS_WIN, 0, 1, _nop)
    for label, key, mx, scale in TRACKBARS:
        init = int(round(float(params[key]) / scale))
        init = max(0, min(mx, init))
        cv2.createTrackbar(label, CONTROLS_WIN, init, mx, _nop)


def _read_pause():
    try:
        return cv2.getTrackbarPos(BUTTON_PAUSE, CONTROLS_WIN) == 1
    except cv2.error:
        return False


def _set_pause(on):
    try:
        cv2.setTrackbarPos(BUTTON_PAUSE, CONTROLS_WIN, 1 if on else 0)
    except cv2.error:
        pass


def _read_controls(params):
    for label, key, _mx, scale in TRACKBARS:
        v = cv2.getTrackbarPos(label, CONTROLS_WIN)
        params[key] = v * scale if scale != 1.0 else int(v)
    return params


def _set_controls(params):
    for label, key, _mx, scale in TRACKBARS:
        v = int(round(float(params[key]) / scale))
        cv2.setTrackbarPos(label, CONTROLS_WIN, v)


def _show_diag(diag, steer_deg, mode):
    # tile 320x180 thumbnails; main "8_overlay" stays larger
    thumb_size = (320, 180)
    for name, img in diag.items():
        if name == "8_overlay":
            shown = img.copy()
            cv2.putText(shown, f"Steer: {steer_deg:+.2f} deg", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(shown, f"Mode:  {mode}", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow(name, shown)
        else:
            cv2.imshow(name, cv2.resize(img, thumb_size))


def calibration_loop(cap_or_img, roi_norm, params, source_kind):
    """source_kind: 'video' | 'camera' | 'image'."""
    _build_controls(params)
    print("\n=== CALIBRATION ===")
    print("  SPACE / 'Pause' trackbar  freeze on current frame")
    print("  n  next frame             p  prev frame (video)")
    print("  s  save preset            r  reload preset       a  re-run autotune")
    print("  g  GO (run main loop)     ESC quit\n")

    paused_keyboard = (source_kind == "image")
    last_frame = None
    if source_kind == "image":
        last_frame = cap_or_img
        _set_pause(True)   # show pause button as active for the user

    while True:
        # Pause is the OR of the keyboard-toggle and the trackbar button
        paused = paused_keyboard or _read_pause()

        if not paused:
            if source_kind == "image":
                frame = last_frame
            else:
                ret, frame = cap_or_img.read()
                if not ret:
                    if source_kind == "video":
                        cap_or_img.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                last_frame = frame
        else:
            frame = last_frame

        if frame is None:
            break

        params = _read_controls(params)
        result, closed, dbg, steer_deg, mode, diag = pipeline(frame, roi_norm, params, debug=True)
        _show_diag(diag, steer_deg, mode)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:                            # ESC
            return None
        elif key == ord(' '):
            paused_keyboard = not paused_keyboard
            # Mirror the new state on the trackbar so the user sees it
            _set_pause(paused_keyboard or _read_pause())
        elif key == ord('n') and paused and source_kind != "image":
            ret, fr = cap_or_img.read()
            if ret:
                last_frame = fr
        elif key == ord('p') and paused and source_kind == "video":
            pos = int(cap_or_img.get(cv2.CAP_PROP_POS_FRAMES))
            cap_or_img.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
            ret, fr = cap_or_img.read()
            if ret:
                last_frame = fr
        elif key == ord('s'):
            save_params(params)
        elif key == ord('r'):
            params = load_params()
            _set_controls(params)
        elif key == ord('a'):
            # Re-run autotune over the source. Rewinds video to start.
            print("Re-running autotune...")
            params = autotune_params(cap_or_img, roi_norm, source_kind)
            _set_controls(params)
            save_params(params)
            if source_kind == "video":
                cap_or_img.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif key == ord('g'):
            save_params(params)
            # close diag windows so the main loop has a clean screen
            for n in list(diag.keys()):
                try:
                    cv2.destroyWindow(n)
                except cv2.error:
                    pass
            try:
                cv2.destroyWindow(CONTROLS_WIN)
            except cv2.error:
                pass
            return params


# =====================================================================
# MAIN DETECTION LOOP
# =====================================================================
def _run_loop(cap, roi_norm, params, source_kind):
    prev_t = time.time()
    fps_avg = 0.0
    while True:
        if source_kind == "image":
            frame = cap
        else:
            ret, frame = cap.read()
            if not ret:
                break

        now = time.time()
        fps = 1.0 / (now - prev_t + 1e-5)
        prev_t = now
        fps_avg = (fps_avg * 0.9) + (fps * 0.1) if fps_avg > 0 else fps

        result, closed, dbg, steer_deg, mode = pipeline(frame, roi_norm, params, debug=False)

        cv2.putText(result, f"Steering: {steer_deg:+.2f} deg", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"Mode: {mode}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(result, f"FPS:  {int(fps_avg)}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)

        cv2.imshow("Final Output",   result)
        cv2.imshow("Bird's Eye Mask", closed)

        key = cv2.waitKey(1 if source_kind != "image" else 0) & 0xFF
        if key == 27:
            return False
        if key == ord('c'):
            return True   # signal: re-enter calibration

        if source_kind == "image":
            return False


# =====================================================================
# ENTRY POINTS
# =====================================================================
def _calibrate_then_run(cap_or_img, roi_norm, source_kind, force_autotune=False):
    # If no preset file exists, OR caller asked, run autotune first.
    if force_autotune or not os.path.exists(TUNE_FILE):
        params = autotune_params(cap_or_img, roi_norm, source_kind)
        save_params(params)
        if source_kind == "video":
            cap_or_img.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        params = load_params()

    while True:
        if source_kind == "image":
            params2 = calibration_loop(cap_or_img, roi_norm, params, "image")
        else:
            params2 = calibration_loop(cap_or_img, roi_norm, params, source_kind)
        if params2 is None:
            return
        params = params2
        reset_state()
        again = _run_loop(cap_or_img, roi_norm, params, source_kind)
        if not again:
            return
        # 'c' pressed: rewind video so calibration sees real footage
        if source_kind == "video":
            cap_or_img.set(cv2.CAP_PROP_POS_FRAMES, 0)


def process_video(video_path, force_reselect=False, autotune=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return
    ret, first = cap.read()
    if not ret:
        cap.release(); return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    roi_norm = select_and_save_roi(first) if force_reselect else load_roi(first)
    if roi_norm is None:
        cap.release(); return

    _calibrate_then_run(cap, roi_norm, "video", force_autotune=autotune)
    cap.release()
    cv2.destroyAllWindows()


def process_camera(camera_index=0, force_reselect=False, autotune=False):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, first = cap.read()
    if not ret:
        cap.release(); return

    print(f"Camera at {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    roi_norm = select_and_save_roi(first) if force_reselect else load_roi(first)
    if roi_norm is None:
        cap.release(); return

    _calibrate_then_run(cap, roi_norm, "camera", force_autotune=autotune)
    cap.release()
    cv2.destroyAllWindows()


def process_image(path, force_reselect=False, autotune=False):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read {path}"); return
    roi_norm = select_and_save_roi(img) if force_reselect else load_roi(img)
    if roi_norm is None:
        return
    _calibrate_then_run(img, roi_norm, "image", force_autotune=autotune)
    cv2.destroyAllWindows()


# -------- RUN --------
if __name__ == "__main__":
    # First time on a new video / new lighting: pass autotune=True so the
    # algorithm samples the source and computes thresholds that already
    # detect the lines. After that, tune_v8.json holds the values.
    process_video("data base/WIN_20260407_12_47_46_Pro.mp4", autotune=False)
