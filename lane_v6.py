import cv2
import numpy as np
import os
import time

ROI_FILE = "roi_v6.npy"

# ---- FIXED SIZE ----
TARGET_WIDTH = 480
TARGET_HEIGHT = 270

clicked_points = []
clone_img = None
smoothed_center_fit = None

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

    # WINDOW_NORMAL allows you to resize the window if the raw camera resolution is larger than your screen
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
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
        img_h, img_w = img.shape[:2]

        # normalize (0->1)
        pts[:, 0] /= img_w
        pts[:, 1] /= img_h

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
    global smoothed_center_fit
    img = resize_image(frame)
    height, width = img.shape[:2]

    # Convert normalized ROI back to pixel coords
    src_points = roi_norm.copy()
    src_points[:, 0] *= width
    src_points[:, 1] *= height

    horizon_y = int(min(src_points[0][1], src_points[1][1]))

    # --- 1. Dynamic Black Color Filter (Enhanced V6) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # This heavily boosts the contrast of lane lines against the road.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    
    # OPTIMIZATION: Only process the road area below the horizon.
    gray_road = gray_clahe[horizon_y:height, :]
    
    # Bilateral Filter replaces Gaussian Blur
    # It preserves sharp edges (lanes) while aggressively blurring flat textures (road asphalt noise)
    blurred_road = cv2.bilateralFilter(gray_road, d=9, sigmaColor=75, sigmaSpace=75)
    
    binary_road = cv2.adaptiveThreshold(
        blurred_road, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        15, 7
    )
    
    # Median Blur wipes out tiny "salt and pepper" white noise dots
    binary_road = cv2.medianBlur(binary_road, 3)

    # --- EXPLICIT STRICT BLACK COLOR FILTER ---
    # Adaptive threshold finds anything "darker than its surroundings" (which accidentally includes shadows).
    # We combine it with an HSV filter to strictly isolate pixels that are truly BLACK.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_road = hsv[horizon_y:height, :]
    
    lower_black = np.array([0, 0, 0])
    # Relaxed from 70 to 110: Allows lines hit by glare/sunlight to still be detected
    upper_black = np.array([180, 255, 110]) 
    black_mask = cv2.inRange(hsv_road, lower_black, upper_black)
    
    # Combine both: Must have a crisp edge (adaptive) AND be actually black (HSV)
    combined_binary = cv2.bitwise_and(binary_road, black_mask)

    # --- 2. Apply Wide ROI Mask (Optimized) ---
    masked_binary = np.zeros_like(gray)
    masked_binary[horizon_y:height, :] = combined_binary

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

    # --- 4. Morphological Cleaning ---
    # Morphological Open (Erosion then Dilation) to completely delete small floating noise blobs
    open_kernel = np.ones((3, 3), np.uint8)
    warped_cleaned = cv2.morphologyEx(warped, cv2.MORPH_OPEN, open_kernel)
    
    # Increased kernel size heavily (25, 5). This acts as a vertical bridge,
    # melting disconnected dashed lines together so they pass the strict shape filter!
    close_kernel = np.ones((25, 5), np.uint8) 
    closed = cv2.morphologyEx(warped_cleaned, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_warp = np.zeros((height, width, 3), dtype=np.uint8)

    left_fits = []
    right_fits = []

    mask_c = np.zeros_like(warped)

    # --- 5. Shape Filtering ---
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        aspect_ratio = float(h) / w if w > 0 else 0

        # Slightly relaxed area/height since the dashed lines are now being fused together
        if area > 100 and h > height * 0.20 and aspect_ratio > 2.0:
            mask_c.fill(0)
            cv2.drawContours(mask_c, [cnt], -1, 255, -1)

            ys_local, xs_local = np.nonzero(mask_c[y:y+h, x:x+w])
            ys, xs = ys_local + y, xs_local + x

            if len(xs) > 40:
                fit = np.polyfit(ys, xs, 2)
                
                bottom_y = np.max(ys)
                x_bottom = fit[0]*(bottom_y**2) + fit[1]*bottom_y + fit[2]

                if x_bottom < width * 0.10 or x_bottom > width * 0.90:
                    continue

                if x_bottom < width / 2:
                    left_fits.append((fit, x_bottom, ys))
                else:
                    right_fits.append((fit, x_bottom, ys))

    # --- 6. Steering Calculation & Lane Rebuild ---
    steer_deg = 0.0
    rebuild_mode = "No Lanes"

    left_fit = None
    if len(left_fits) > 0:
        left_fits.sort(key=lambda item: item[1], reverse=True)
        left_fit, _, ys_left = left_fits[0]
        
        ploty = np.linspace(np.min(ys_left), np.max(ys_left), 100)
        fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        pts = np.array([np.transpose(np.vstack([fitx, ploty]))])
        cv2.polylines(color_warp, np.int32([pts]), False, (255, 0, 0), 20)

    right_fit = None
    if len(right_fits) > 0:
        right_fits.sort(key=lambda item: item[1], reverse=False)
        right_fit, _, ys_right = right_fits[0]
        
        ploty = np.linspace(np.min(ys_right), np.max(ys_right), 100)
        fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts = np.array([np.transpose(np.vstack([fitx, ploty]))])
        cv2.polylines(color_warp, np.int32([pts]), False, (0, 0, 255), 20)

    center_fit = None

    if left_fit is not None and right_fit is not None:
        center_fit = (left_fit + right_fit) / 2
        rebuild_mode = "2 Lanes Rebuild"
    elif left_fit is not None:
        center_fit = left_fit.copy()
        center_fit[2] += width * 0.25
        rebuild_mode = "1 Lane Rebuild (Left)"
    elif right_fit is not None:
        center_fit = right_fit.copy()
        center_fit[2] -= width * 0.25
        rebuild_mode = "1 Lane Rebuild (Right)"

    # --- 7. Temporal Smoothing Filter ---
    if center_fit is not None:
        if smoothed_center_fit is None:
            smoothed_center_fit = center_fit
        else:
            alpha = 0.2 
            smoothed_center_fit = alpha * center_fit + (1 - alpha) * smoothed_center_fit
    elif smoothed_center_fit is not None:
        rebuild_mode = "History Fallback"

    if smoothed_center_fit is not None:
        ploty = np.linspace(0, height-1, height)
        center_x = smoothed_center_fit[0]*ploty**2 + smoothed_center_fit[1]*ploty + smoothed_center_fit[2]

        pts_center = np.array([np.transpose(np.vstack([center_x, ploty]))])
        cv2.polylines(color_warp, np.int32([pts_center]), False, (0, 255, 0), 100)

        lookahead_y = height * 0.7
        target_x = smoothed_center_fit[0]*(lookahead_y**2) + smoothed_center_fit[1]*lookahead_y + smoothed_center_fit[2]

        car_center_x = width / 2.0
        cte = target_x - car_center_x

        slope = 2 * smoothed_center_fit[0] * lookahead_y + smoothed_center_fit[1]
        heading_error_deg = np.degrees(np.arctan(slope))

        Kp = 0.15 
        Kd = 0.40 

        steer_deg = (Kp * cte) + (Kd * heading_error_deg)
        steer_deg = np.clip(steer_deg, -30, 30)

        cv2.circle(color_warp, (int(target_x), int(lookahead_y)), 15, (0, 165, 255), -1)

    # Overlay
    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))
    result = cv2.addWeighted(img, 1, newwarp, 0.8, 0)

    return result, closed, steer_deg, rebuild_mode

# ---------- CAMERA PROCESSOR ----------
def process_camera(camera_index=0, force_reselect=False):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    # Attempt to set camera resolution to a higher value for better ROI selection clarity
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Read first frame to assign ROI
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        cap.release() 
        return

    # Print the actual resolution the camera is operating at
    print(f"Camera opened at resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    if first_frame.shape[1] < 600:
        print("Warning: Camera resolution is still low. ROI selection might be less precise than ideal.")

    # Pass the full original frame to the ROI selector so the window is large and clear
    roi_norm = select_and_save_roi(first_frame) if force_reselect else load_roi(first_frame)
    if roi_norm is None: return

    prev_time = time.time()
    fps_avg = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break # End of live stream

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-5)
        prev_time = current_time
        fps_avg = (fps_avg * 0.9) + (fps * 0.1) if fps_avg > 0 else fps

        result, binary_view, steer_deg, rebuild_mode = pipeline(frame, roi_norm)

        # Display Overlay Data
        cv2.putText(result, f"Steering: {steer_deg:.2f} deg", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"Mode: {rebuild_mode}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(result, f"FPS: {int(fps_avg)}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)

        cv2.imshow("Final Output", result)
        cv2.imshow("Bird's Eye Mask", binary_view)

        if cv2.waitKey(1) & 0xFF == 27: # Press Esc to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# -------- RUN EXAMPLES --------
if __name__ == "__main__":
    # Process live video from the USB camera
    process_camera(2)