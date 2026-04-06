import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

ROI_FILE = "roi.npy"

# ---- FIXED SIZE ----
TARGET_WIDTH = 960
TARGET_HEIGHT = 540

clicked_points = []
clone_img = None

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

# ---------- ROI SAVE (NORMALIZED) ----------
def select_and_save_roi(image_path):
    global clicked_points, clone_img
    clicked_points = []

    img = cv2.imread(image_path)
    img = resize_image(img)
    clone_img = img.copy()

    cv2.imshow("Select ROI (TL→TR→BR→BL)", clone_img)
    cv2.setMouseCallback("Select ROI (TL→TR→BR→BL)", mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(clicked_points) == 4:
            break
        elif key == 27:
            break

    cv2.destroyAllWindows()

    if len(clicked_points) == 4:
        pts = np.array(clicked_points, dtype=np.float32)

        # normalize (0→1)
        pts[:, 0] /= TARGET_WIDTH
        pts[:, 1] /= TARGET_HEIGHT

        np.save(ROI_FILE, pts)
        print("ROI saved (normalized).")
        return pts
    else:
        return None

# ---------- LOAD ROI ----------
def load_roi(image_path):
    if os.path.exists(ROI_FILE):
        pts = np.load(ROI_FILE)
        print("Loaded ROI.")
        return pts
    else:
        print("No ROI found. Select it now.")
        return select_and_save_roi(image_path)

# ---------- MAIN ----------
def process_lane_image(image_path, force_reselect=False):

    if force_reselect:
        roi_norm = select_and_save_roi(image_path)
    else:
        roi_norm = load_roi(image_path)

    if roi_norm is None:
        return

    img = cv2.imread(image_path)
    img = resize_image(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    # convert normalized ROI → pixel
    src_points = roi_norm.copy()
    src_points[:, 0] *= width
    src_points[:, 1] *= height

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    mask = np.zeros_like(binary_mask)
    cv2.fillPoly(mask, np.int32([src_points]), 255)
    masked_binary = cv2.bitwise_and(binary_mask, mask)

    dst_points = np.float32([
        [width * 0.25, 0],
        [width * 0.75, 0],
        [width * 0.75, height],
        [width * 0.25, height]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)

    warped = cv2.warpPerspective(masked_binary, matrix, (width, height))

    kernel = np.ones((50, 5), np.uint8)
    closed = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_warp = np.zeros((height, width, 3), dtype=np.uint8)

    left_fits = []
    right_fits = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if cv2.contourArea(cnt) > 150 and h > height * 0.25:
            mask_c = np.zeros_like(warped)
            cv2.drawContours(mask_c, [cnt], -1, 255, -1)

            ys, xs = np.nonzero(mask_c)

            if len(xs) > 100:
                fit = np.polyfit(ys, xs, 2)

                if np.mean(xs) < width / 2:
                    left_fits.append(fit)
                    color = (255, 0, 0)
                else:
                    right_fits.append(fit)
                    color = (0, 0, 255)

                ploty = np.linspace(np.min(ys), np.max(ys), 100)
                fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

                pts = np.array([np.transpose(np.vstack([fitx, ploty]))])
                cv2.polylines(color_warp, np.int32([pts]), False, color, 20)

    if len(left_fits) > 0 and len(right_fits) > 0:
        center_fit = (np.mean(left_fits, axis=0) + np.mean(right_fits, axis=0)) / 2

        ploty = np.linspace(0, height-1, height)
        center_x = center_fit[0]*ploty**2 + center_fit[1]*ploty + center_fit[2]

        pts_center = np.array([np.transpose(np.vstack([center_x, ploty]))])
        cv2.polylines(color_warp, np.int32([pts_center]), False, (0,255,0), 200)

        # steering
        y_eval = height
        slope = 2*center_fit[0]*y_eval + center_fit[1]
        steer_deg = np.degrees(np.arctan(slope))
        steer_deg = np.clip(steer_deg, -30, 30)

        print(f"Steering: {steer_deg:.2f} deg")

    else:
        print("Lane detection failed")

    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))
    result = cv2.addWeighted(img_rgb, 1, newwarp, 0.8, 0)

    plt.imshow(result)
    plt.title("Final Output")
    plt.axis('off')
    plt.show()


# -------- RUN --------
process_lane_image("4.png")

# force reselect ROI:
# process_lane_image("2.png", force_reselect=True)