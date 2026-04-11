import cv2
import numpy as np
import os

ROI_FILE = "roi_v4.npy"

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
def select_and_save_roi(img):
    global clicked_points, clone_img
    clicked_points = []

    clone_img = img.copy()

    cv2.imshow("Select ROI", clone_img)
    print("Please click 4 points for ROI (TL -> TR -> BR -> BL). Press ESC to cancel.")
    cv2.setMouseCallback("Select ROI", mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(clicked_points) == 4:
            cv2.waitKey(500) # slight delay so user can see last line
            break
        elif key == 27:
            break

    cv2.destroyWindow("Select ROI")

    if len(clicked_points) == 4:
        pts = np.array(clicked_points, dtype=np.float32)

        # normalize (0->1)
        pts[:, 0] /= TARGET_WIDTH
        pts[:, 1] /= TARGET_HEIGHT

        np.save(ROI_FILE, pts)
        print("ROI saved (normalized).")
        return pts
    else:
        return None

# ---------- LOAD ROI ----------
def load_roi(img):
    if os.path.exists(ROI_FILE):
        pts = np.load(ROI_FILE)
        print("Loaded existing ROI.")
        return pts
    else:
        print("No ROI found. Please select it now.")
        return select_and_save_roi(img)

# ---------- MAIN PIPELINE ----------
def pipeline(frame, roi_norm):
    img = resize_image(frame)
    height, width = img.shape[:2]

    # Convert normalized ROI back to pixel coords
    src_points = roi_norm.copy()
    src_points[:, 0] *= width
    src_points[:, 1] *= height

    # --- 1. Dynamic Black Color Filter ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Adaptive Threshold handles changing lighting and shadows naturally.
    # THRESH_BINARY_INV isolates dark lanes by turning dark regions white.
    binary_mask = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        31, 15
    )

    # --- 2. Apply ROI Mask ---
    mask = np.zeros_like(binary_mask)
    cv2.fillPoly(mask, np.int32([src_points]), 255)
    masked_binary = cv2.bitwise_and(binary_mask, mask)

    # --- 3. Perspective Transform ---
    dst_points = np.float32([
        [width * 0.25, 0],
        [width * 0.75, 0],
        [width * 0.75, height],
        [width * 0.25, height]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)

    warped = cv2.warpPerspective(masked_binary, matrix, (width, height))

    # --- 4. Morphological Close (Combine components focusing on vertical lines) ---
    kernel = np.ones((25, 5), np.uint8) 
    closed = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_warp = np.zeros((height, width, 3), dtype=np.uint8)

    left_fits = []
    right_fits = []

    # --- 5. Shape Filtering ---
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Calculate Aspect Ratio -> we only want components that look like a line (taller than they are wide)
        aspect_ratio = float(h) / w if w > 0 else 0

        # Filter: Area size, relative height, and Aspect Ratio (Shape Constraint)
        if area > 100 and h > height * 0.15 and aspect_ratio > 1.2:
            mask_c = np.zeros_like(warped)
            cv2.drawContours(mask_c, [cnt], -1, 255, -1)

            ys, xs = np.nonzero(mask_c)

            if len(xs) > 50:
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

    # --- 6. Steering Calculation ---
    steer_deg = 0.0
    if len(left_fits) > 0 and len(right_fits) > 0:
        center_fit = (np.mean(left_fits, axis=0) + np.mean(right_fits, axis=0)) / 2

        ploty = np.linspace(0, height-1, height)
        center_x = center_fit[0]*ploty**2 + center_fit[1]*ploty + center_fit[2]

        pts_center = np.array([np.transpose(np.vstack([center_x, ploty]))])
        cv2.polylines(color_warp, np.int32([pts_center]), False, (0, 255, 0), 100)

        y_eval = height
        slope = 2 * center_fit[0] * y_eval + center_fit[1]
        steer_deg = np.degrees(np.arctan(slope))
        steer_deg = np.clip(steer_deg, -30, 30)

    # Overlay
    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))
    result = cv2.addWeighted(img, 1, newwarp, 0.8, 0)

    return result, closed, steer_deg

# ---------- VIDEO PROCESSOR ----------
def process_video(video_path, force_reselect=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Read first frame to assign ROI
    ret, first_frame = cap.read()
    if not ret: return

    first_frame_resized = resize_image(first_frame)
    roi_norm = select_and_save_roi(first_frame_resized) if force_reselect else load_roi(first_frame_resized)
    if roi_norm is None: return

    while True:
        ret, frame = cap.read()
        if not ret: break # End of video

        result, binary_view, steer_deg = pipeline(frame, roi_norm)

        # Display Overlay Data
        cv2.putText(result, f"Steering: {steer_deg:.2f} deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Final Output", result)
        cv2.imshow("Bird's Eye Mask", binary_view)

        if cv2.waitKey(25) & 0xFF == 27: # Press Esc to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# -------- RUN EXAMPLES --------
if __name__ == "__main__":
    # Example for Video Input:
    process_video("data base/WIN_20260407_12_47_46_Pro.mp4")
    
    # Since you initially had images, you can test video streams like this:
    print("To test video uncomment process_video('your_video.mp4') at the bottom of the script.")