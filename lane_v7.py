import cv2
import numpy as np
import os
import time

# =====================================================================
# Lane v7 - Stable detector for black tape on near-white floor
#
# Key changes vs v6:
#   * Mask = black-hat(L) AND HSV-black(low S, low V) AND Lab-dark
#     -> rejects golden marble veins (high S) and reflections (high V),
#        and suppresses large dark blobs like the chassis.
#   * Narrower close kernel (15,3) so 2 close parallel lines do not merge.
#   * Per-side tracking with a position gate -> no left/right flip.
#   * When >2 lines are present: pick the LEFT NEAREST and RIGHT NEAREST
#     lines to the car at a near-car reference row.
#   * Adaptive lane half-width learned from frames where both sides seen.
# =====================================================================

ROI_FILE = "roi_v7.npy"

# ---- FIXED SIZE ----
TARGET_WIDTH = 480
TARGET_HEIGHT = 270

# ---- TUNING ----
T_BLACKHAT      = 5       # min black-hat response to count as a line pixel
HSV_V_MAX       = 100       # value <= this  (true black, not gray)
HSV_S_MAX       = 120       # saturation <= this  (drops golden marble veins)
LAB_L_MAX       = 105      # absolute lightness ceiling

CLOSE_KERNEL    = (15, 3)  # vertical bridge for dashed lines, narrow x
OPEN_KERNEL     = (3, 3)

MIN_AREA        = 60
MIN_H_FRAC      = 0.10     # min contour height as fraction of warped height
MIN_ASPECT      = 1.4      # h / w
MAX_ASPECT_HORZ = 0.6      # reject near-horizontal blobs (h / w too small)

Y_REF_FRAC      = 0.85     # row where we evaluate "nearest to car"
LOOKAHEAD_FRAC  = 0.65
DEAD_BAND_PX    = 8        # ignore candidates within +/- this of car center
TRACK_GATE_PX   = 70       # max jump in x_ref between frames per side
LOSE_FRAMES     = 8        # how many frames to coast on a lost side
DEFAULT_HALF_W  = 0.25     # initial half-lane width as fraction of width

EMA_ALPHA       = 0.20

KP, KD          = 0.15, 0.40

# ---- STATE ----
clicked_points = []
clone_img = None

class TrackState:
    def __init__(self):
        self.prev_left_fit  = None
        self.prev_right_fit = None
        self.prev_left_x    = None
        self.prev_right_x   = None
        self.miss_left      = 0
        self.miss_right     = 0
        self.half_width_px  = TARGET_WIDTH * DEFAULT_HALF_W
        self.smoothed_center_fit = None
        self.show_debug     = True

state = TrackState()


def resize_image(img):
    return cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))


# ---------- MOUSE ----------
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
    print("Please click 4 points for ROI (TL -> TR -> BR -> BL). Press ESC to cancel.")
    cv2.setMouseCallback("Select ROI", mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(clicked_points) == 4:
            cv2.waitKey(500)
            break
        elif key == 27:
            break

    cv2.destroyWindow("Select ROI")

    if len(clicked_points) == 4:
        pts = np.array(clicked_points, dtype=np.float32)
        img_h, img_w = img.shape[:2]
        pts[:, 0] /= img_w
        pts[:, 1] /= img_h
        np.save(ROI_FILE, pts)
        print("ROI saved (normalized).")
        return pts
    return None


def load_roi(img):
    if os.path.exists(ROI_FILE):
        pts = np.load(ROI_FILE)
        print("Loaded existing ROI.")
        return pts
    print("No ROI found. Please select it now.")
    return select_and_save_roi(img)


# ---------- MASK ----------
def build_black_mask(img, horizon_y):
    """Return a uint8 binary mask, full image size, lit only where a black
    tape pixel is highly likely. Combines three independent gates."""
    height, width = img.shape[:2]

    # Lab L channel - cleaner than gray on cream marble
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # Operate only on the road region for speed
    L_road = L[horizon_y:height, :]

    # Gentle smoothing to kill speckle but keep tape edges
    L_road_blur = cv2.GaussianBlur(L_road, (5, 5), 0)

    # Black-hat: bright where there is a thin DARK structure on a bright field.
    # A horizontal kernel ensures it responds to vertical/curving tape.
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blackhat = cv2.morphologyEx(L_road_blur, cv2.MORPH_BLACKHAT, bh_kernel)
    _, bh_mask = cv2.threshold(blackhat, T_BLACKHAT, 255, cv2.THRESH_BINARY)

    # HSV gate -> drops golden marble veins (high S) and reflections (high V)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_road = hsv[horizon_y:height, :]
    lower = np.array([0, 0, 0], dtype=np.uint8)
    upper = np.array([180, HSV_S_MAX, HSV_V_MAX], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv_road, lower, upper)

    # Absolute Lab L gate
    _, lab_mask = cv2.threshold(L_road, LAB_L_MAX, 255, cv2.THRESH_BINARY_INV)

    combined = cv2.bitwise_and(bh_mask, hsv_mask)
    combined = cv2.bitwise_and(combined, lab_mask)

    # Light despeckle
    combined = cv2.medianBlur(combined, 3)

    full = np.zeros((height, width), dtype=np.uint8)
    full[horizon_y:height, :] = combined
    return full


# ---------- CONTOUR -> CANDIDATE FIT ----------
def fit_candidates(closed, height, width):
    """Return list of candidates: dict(fit, x_ref, ys, xs)."""
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cands = []
    y_ref = height * Y_REF_FRAC

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        if h < height * MIN_H_FRAC:
            continue
        if w == 0:
            continue
        aspect = float(h) / float(w)
        if aspect < MIN_ASPECT:
            continue

        pts = cnt.reshape(-1, 2)
        xs = pts[:, 0].astype(np.float32)
        ys = pts[:, 1].astype(np.float32)
        if len(xs) < 25:
            continue

        # 2nd-order polyfit x = a y^2 + b y + c
        try:
            fit = np.polyfit(ys, xs, 2)
        except np.linalg.LinAlgError:
            continue

        # residual RMS - reject scattered blobs
        pred = fit[0] * ys * ys + fit[1] * ys + fit[2]
        rms = float(np.sqrt(np.mean((pred - xs) ** 2)))
        if rms > 12.0:
            continue

        x_ref = float(fit[0] * y_ref * y_ref + fit[1] * y_ref + fit[2])
        if x_ref < width * 0.05 or x_ref > width * 0.95:
            continue

        cands.append({
            "fit": fit,
            "x_ref": x_ref,
            "y_min": float(np.min(ys)),
            "y_max": float(np.max(ys)),
            "rms": rms,
        })
    return cands


# ---------- ASSIGN LEFT / RIGHT NEAREST ----------
def pick_nearest(cands, car_x, prev_x, miss_count):
    """For one side, return (best_fit, best_x_ref, miss_count_new) or
    (None, None, miss_count_new) if no candidate qualifies. Caller sets
    side via the candidate list it passes in (already filtered)."""
    if not cands:
        return None, None, miss_count + 1

    if prev_x is not None:
        # Prefer candidates within tracking gate; if any exist, restrict.
        gated = [c for c in cands if abs(c["x_ref"] - prev_x) <= TRACK_GATE_PX]
        if gated:
            cands = gated

    # nearest to car_x
    best = min(cands, key=lambda c: abs(c["x_ref"] - car_x))
    return best["fit"], best["x_ref"], 0


# ---------- MAIN PIPELINE ----------
def pipeline(frame, roi_norm):
    img = resize_image(frame)
    height, width = img.shape[:2]

    src_points = roi_norm.copy().astype(np.float32)
    src_points[:, 0] *= width
    src_points[:, 1] *= height

    horizon_y = int(min(src_points[0][1], src_points[1][1]))

    # 1. Black-pixel mask
    masked_binary = build_black_mask(img, horizon_y)

    # 2. Perspective warp to bird's eye
    dst_points = np.float32([
        [width * 0.25, 0],
        [width * 0.75, 0],
        [width * 0.75, height],
        [width * 0.25, height],
    ])
    M    = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped = cv2.warpPerspective(masked_binary, M, (width, height))

    # 3. Morph clean - narrow close so adjacent parallel lines stay separate
    open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_KERNEL)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL)
    cleaned = cv2.morphologyEx(warped, cv2.MORPH_OPEN, open_k)
    closed  = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_k)

    # 4. Candidate fits
    cands = fit_candidates(closed, height, width)

    # 5. Split by car center, pick nearest on each side
    car_x = width / 2.0
    left_pool  = [c for c in cands if c["x_ref"] < car_x - DEAD_BAND_PX]
    right_pool = [c for c in cands if c["x_ref"] > car_x + DEAD_BAND_PX]

    left_fit,  left_x,  state.miss_left  = pick_nearest(
        left_pool,  car_x, state.prev_left_x,  state.miss_left)
    right_fit, right_x, state.miss_right = pick_nearest(
        right_pool, car_x, state.prev_right_x, state.miss_right)

    # 6. Per-side history fallback (coast a few frames if lost)
    if left_fit is None and state.prev_left_fit is not None and state.miss_left <= LOSE_FRAMES:
        left_fit = state.prev_left_fit
        left_x   = state.prev_left_x
    if right_fit is None and state.prev_right_fit is not None and state.miss_right <= LOSE_FRAMES:
        right_fit = state.prev_right_fit
        right_x   = state.prev_right_x

    # Update prev only on fresh detections
    if state.miss_left == 0 and left_fit is not None:
        state.prev_left_fit = left_fit
        state.prev_left_x   = left_x
    if state.miss_right == 0 and right_fit is not None:
        state.prev_right_fit = right_fit
        state.prev_right_x   = right_x

    # 7. Adaptive lane half-width when both fresh
    if state.miss_left == 0 and state.miss_right == 0 and left_x is not None and right_x is not None:
        measured = (right_x - left_x) / 2.0
        if measured > 30:  # sanity
            state.half_width_px = 0.8 * state.half_width_px + 0.2 * measured

    # 8. Center fit
    rebuild_mode = "No Lanes"
    center_fit = None
    if left_fit is not None and right_fit is not None:
        center_fit = (left_fit + right_fit) / 2.0
        rebuild_mode = "2 Lanes" if (state.miss_left == 0 and state.miss_right == 0) else "2 Lanes (coast)"
    elif left_fit is not None:
        center_fit = left_fit.copy()
        center_fit[2] += state.half_width_px
        rebuild_mode = "1 Lane (Left)"
    elif right_fit is not None:
        center_fit = right_fit.copy()
        center_fit[2] -= state.half_width_px
        rebuild_mode = "1 Lane (Right)"

    if center_fit is not None:
        if state.smoothed_center_fit is None:
            state.smoothed_center_fit = center_fit
        else:
            state.smoothed_center_fit = (
                EMA_ALPHA * center_fit + (1 - EMA_ALPHA) * state.smoothed_center_fit)
    elif state.smoothed_center_fit is not None:
        rebuild_mode = "History Fallback"

    # 9. Draw
    color_warp = np.zeros((height, width, 3), dtype=np.uint8)

    # all candidates faint grey
    if state.show_debug:
        for c in cands:
            f = c["fit"]
            ploty = np.linspace(c["y_min"], c["y_max"], 80)
            fitx = f[0] * ploty * ploty + f[1] * ploty + f[2]
            pts = np.array([np.transpose(np.vstack([fitx, ploty]))], dtype=np.int32)
            cv2.polylines(color_warp, pts, False, (90, 90, 90), 4)

    if left_fit is not None:
        ploty = np.linspace(0, height - 1, height)
        fitx = left_fit[0] * ploty * ploty + left_fit[1] * ploty + left_fit[2]
        pts = np.array([np.transpose(np.vstack([fitx, ploty]))], dtype=np.int32)
        cv2.polylines(color_warp, pts, False, (255, 0, 0), 14)

    if right_fit is not None:
        ploty = np.linspace(0, height - 1, height)
        fitx = right_fit[0] * ploty * ploty + right_fit[1] * ploty + right_fit[2]
        pts = np.array([np.transpose(np.vstack([fitx, ploty]))], dtype=np.int32)
        cv2.polylines(color_warp, pts, False, (0, 0, 255), 14)

    steer_deg = 0.0
    if state.smoothed_center_fit is not None:
        f = state.smoothed_center_fit
        ploty = np.linspace(0, height - 1, height)
        cx = f[0] * ploty * ploty + f[1] * ploty + f[2]
        pts_c = np.array([np.transpose(np.vstack([cx, ploty]))], dtype=np.int32)
        cv2.polylines(color_warp, pts_c, False, (0, 255, 0), 60)

        lookahead_y = height * LOOKAHEAD_FRAC
        target_x = f[0] * lookahead_y * lookahead_y + f[1] * lookahead_y + f[2]
        cte = target_x - car_x
        slope = 2 * f[0] * lookahead_y + f[1]
        heading_err_deg = np.degrees(np.arctan(slope))
        steer_deg = float(np.clip(KP * cte + KD * heading_err_deg, -30, 30))
        cv2.circle(color_warp, (int(target_x), int(lookahead_y)), 12, (0, 165, 255), -1)

    # Overlay back to perspective
    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))
    result = cv2.addWeighted(img, 1, newwarp, 0.8, 0)

    return result, closed, color_warp, steer_deg, rebuild_mode


# ---------- VIDEO ----------
def process_video(video_path, force_reselect=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame from video.")
        cap.release()
        return

    roi_norm = select_and_save_roi(first_frame) if force_reselect else load_roi(first_frame)
    if roi_norm is None:
        return

    _run_loop(cap, roi_norm)
    cap.release()
    cv2.destroyAllWindows()


# ---------- CAMERA ----------
def process_camera(camera_index=0, force_reselect=False):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        cap.release()
        return

    print(f"Camera opened at: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    roi_norm = select_and_save_roi(first_frame) if force_reselect else load_roi(first_frame)
    if roi_norm is None:
        return

    _run_loop(cap, roi_norm)
    cap.release()
    cv2.destroyAllWindows()


# ---------- IMAGE (debug) ----------
def process_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read {path}")
        return
    roi_norm = load_roi(img)
    if roi_norm is None:
        return
    # Reset state for a clean single-frame run
    global state
    state = TrackState()
    result, closed, dbg, steer_deg, mode = pipeline(img, roi_norm)
    cv2.putText(result, f"Steer: {steer_deg:.2f} deg", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(result, f"Mode:  {mode}",              (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("Final Output",   result)
    cv2.imshow("Bird's Eye Mask", closed)
    cv2.imshow("Debug Warp",      dbg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _run_loop(cap, roi_norm):
    prev_time = time.time()
    fps_avg = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-5)
        prev_time = now
        fps_avg = (fps_avg * 0.9) + (fps * 0.1) if fps_avg > 0 else fps

        result, closed, dbg, steer_deg, mode = pipeline(frame, roi_norm)

        cv2.putText(result, f"Steering: {steer_deg:.2f} deg", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"Mode: {mode}",                  (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(result, f"FPS:  {int(fps_avg)}",          (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)

        cv2.imshow("Final Output",   result)
        cv2.imshow("Bird's Eye Mask", closed)
        if state.show_debug:
            cv2.imshow("Debug Warp",  dbg)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:           # ESC
            break
        elif key == ord('d'):   # toggle debug overlay window
            state.show_debug = not state.show_debug
            if not state.show_debug:
                try:
                    cv2.destroyWindow("Debug Warp")
                except cv2.error:
                    pass


# -------- RUN --------
if __name__ == "__main__":
    # Default: process the local video used during development.
    # Switch to process_camera(2) on the robot.
    process_video("data base/WIN_20260407_12_47_46_Pro.mp4")
