"""
Lane v8 - Raspberry Pi launcher.

Reuses the v8 detection pipeline as a library and shows only what is
useful on a small robot screen:

  Window "Final"     - perspective overlay + steering-wheel HUD + FPS
  Window "Mask"      - bird's-eye 'closed' mask (one debug view)
  Window "Controls"  - slim slider set, sized to fit small displays

Keys:
  ESC   quit
  s     save sliders to tune_v8.json
  a     re-run autotune on the live camera, then save

Files used:
  tune_v8.json     <- shared with the desktop tuning workflow
  roi_v8_rpi.npy   <- separate from desktop because the Pi camera POV differs
"""
import cv2
import numpy as np
import time

import lane_v8 as core

# Pi-specific ROI cache so it doesn't clash with the desktop one
core.ROI_FILE = "roi_v8_rpi.npy"

# Drop processing resolution here if the Pi cannot keep up:
# core.TARGET_WIDTH  = 320
# core.TARGET_HEIGHT = 180


CONTROLS_WIN = "Controls"
FINAL_WIN    = "Final"
MASK_WIN     = "Mask"

PI_TRACKBARS = [
    ("T_blackhat",     "T_BLACKHAT",      80,  1.0),
    ("HSV V_max",      "HSV_V_MAX",      255,  1.0),
    ("HSV S_max",      "HSV_S_MAX",      255,  1.0),
    ("Lab L_max",      "LAB_L_MAX",      255,  1.0),
    ("Min area",       "MIN_AREA",       500,  1.0),
    ("Min aspect x10", "MIN_ASPECT",      60,  0.1),
    ("Lane EMA x100",  "LANE_EMA",        90,  0.01),
    ("Hold frames",    "HOLD_FRAMES",     60,  1.0),
    ("Kp x100",        "KP",              60,  0.01),
    ("Kd x100",        "KD",             100,  0.01),
]

PAUSE_BAR = "Pause (0/1)"


def _nop(_): pass


def _build_controls(params):
    cv2.namedWindow(CONTROLS_WIN, cv2.WINDOW_NORMAL)
    # Sized so all 10 sliders + the pause button fit without scrolling.
    cv2.resizeWindow(CONTROLS_WIN, 380, 460)
    cv2.createTrackbar(PAUSE_BAR, CONTROLS_WIN, 0, 1, _nop)
    for label, key, mx, scale in PI_TRACKBARS:
        init = int(round(float(params[key]) / scale))
        init = max(0, min(mx, init))
        cv2.createTrackbar(label, CONTROLS_WIN, init, mx, _nop)


def _read_controls(params):
    for label, key, _mx, scale in PI_TRACKBARS:
        v = cv2.getTrackbarPos(label, CONTROLS_WIN)
        params[key] = v * scale if scale != 1.0 else int(v)
    return params


def _set_controls(params):
    for label, key, _mx, scale in PI_TRACKBARS:
        v = int(round(float(params[key]) / scale))
        try:
            cv2.setTrackbarPos(label, CONTROLS_WIN, v)
        except cv2.error:
            pass


def _draw_steering_wheel(img, steer_deg, cx, cy, R):
    """Wheel rim + 3 spokes rotated by steer_deg + numeric angle."""
    rad = np.radians(steer_deg)
    cv2.circle(img, (cx, cy), R,     (220, 220, 220), 2)
    cv2.circle(img, (cx, cy), R - 6, ( 80,  80,  80), 1)
    for base in (-np.pi / 2.0, np.pi / 6.0, 5.0 * np.pi / 6.0):
        a = base + rad
        ex = int(cx + (R - 4) * np.cos(a))
        ey = int(cy + (R - 4) * np.sin(a))
        cv2.line(img, (cx, cy), (ex, ey), (220, 220, 220), 3)
    cv2.circle(img, (cx, cy), 6, (0, 0, 0), -1)
    cv2.putText(img, f"{steer_deg:+.1f} deg",
                (cx - R, cy + R + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)


def _draw_hud(img, fps, steer_deg, mode):
    h, w = img.shape[:2]
    # FPS plate top-left
    cv2.rectangle(img, (5, 5), (175, 60), (0, 0, 0), -1)
    cv2.putText(img, f"FPS: {fps:5.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"{mode}", (10, 53),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    # Wheel bottom-right
    R = 40
    _draw_steering_wheel(img, steer_deg, w - R - 15, h - R - 30, R)


def run_camera(camera_index=0, force_reselect_roi=False, force_autotune=False):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}")
        return

    # Hint a usable resolution; the cap may pick the closest available.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, first = cap.read()
    if not ret:
        print("Camera grab failed.")
        cap.release()
        return
    print(f"Camera at {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    roi_norm = (core.select_and_save_roi(first)
                if force_reselect_roi else core.load_roi(first))
    if roi_norm is None:
        cap.release()
        return

    # Load preset; autotune if missing or forced
    import os
    if force_autotune or not os.path.exists(core.TUNE_FILE):
        params = core.autotune_params(cap, roi_norm, "camera")
        core.save_params(params)
    else:
        params = core.load_params()

    _build_controls(params)
    core.reset_state()

    prev_t   = time.time()
    fps_avg  = 0.0
    last_fr  = first

    while True:
        params = _read_controls(params)
        paused = cv2.getTrackbarPos(PAUSE_BAR, CONTROLS_WIN) == 1

        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            last_fr = frame
        else:
            frame = last_fr

        result, closed, _warp, steer_deg, mode = core.pipeline(
            frame, roi_norm, params, debug=False)

        now = time.time()
        inst_fps = 1.0 / max(now - prev_t, 1e-3)
        prev_t = now
        fps_avg = (0.9 * fps_avg + 0.1 * inst_fps) if fps_avg > 0 else inst_fps

        _draw_hud(result, fps_avg, steer_deg, mode)

        cv2.imshow(FINAL_WIN, result)
        cv2.imshow(MASK_WIN,  closed)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                 # ESC
            break
        elif key == ord('s'):
            core.save_params(params)
        elif key == ord('a'):
            print("Re-running autotune on live camera...")
            params = core.autotune_params(cap, roi_norm, "camera")
            _set_controls(params)
            core.save_params(params)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Change the index if the robot camera is not /dev/video0.
    run_camera(0)
