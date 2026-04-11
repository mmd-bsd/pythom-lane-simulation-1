import cv2
import numpy as np
import matplotlib.pyplot as plt

clicked_points = []
clone_img = None

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
            
            cv2.imshow("Select ROI - Click 4 points (TL, TR, BR, BL)", clone_img)

def select_roi(image_path):
    global clicked_points, clone_img
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
        
    clone_img = img.copy()
    
    cv2.imshow("Select ROI - Click 4 points (TL, TR, BR, BL)", clone_img)
    cv2.setMouseCallback("Select ROI - Click 4 points (TL, TR, BR, BL)", mouse_callback)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(clicked_points) == 4:
            cv2.waitKey(500) 
            break
        elif key == 27: 
            break
            
    cv2.destroyAllWindows()
    return np.float32(clicked_points)

def process_lane_image(image_path):
    global clicked_points
    clicked_points = [] 
    
    print("Click 4 ROI points: TL → TR → BR → BL")
    src_points = select_roi(image_path)
    
    if src_points is None or len(src_points) != 4:
        print("Error: need 4 points")
        return

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

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
    closed_warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    colors = [(255, 0, 0), (0, 0, 255)]
    
    left_fits = []
    right_fits = []

    print("\n--- Detected Lines ---")

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if cv2.contourArea(contour) > 150 and h > (height * 0.25):
            contour_mask = np.zeros_like(warped)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            line_y, line_x = np.nonzero(contour_mask)

            if len(line_x) > 100:
                fit = np.polyfit(line_y, line_x, 2)

                mean_x = np.mean(line_x)

                if mean_x < width / 2:
                    left_fits.append(fit)
                    color = colors[0]
                else:
                    right_fits.append(fit)
                    color = colors[1]

                ploty = np.linspace(np.min(line_y), np.max(line_y), 100)
                fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

                pts = np.array([np.transpose(np.vstack([fitx, ploty]))])
                cv2.polylines(color_warp, np.int32([pts]), False, color, 20)

    center_fit = None

    if len(left_fits) > 0 and len(right_fits) > 0:
        left_avg = np.mean(left_fits, axis=0)
        right_avg = np.mean(right_fits, axis=0)
        center_fit = (left_avg + right_avg) / 2.0

        print("\n--- Center Line ---")
        print(f"c2={center_fit[0]:.6f}, c1={center_fit[1]:.6f}, c0={center_fit[2]:.2f}")

        ploty = np.linspace(0, height-1, height)
        center_x = center_fit[0]*ploty**2 + center_fit[1]*ploty + center_fit[2]

        pts_center = np.array([np.transpose(np.vstack([center_x, ploty]))])
        cv2.polylines(color_warp, np.int32([pts_center]), False, (0,255,0), 200)

        # Steering
        y_eval = height
        c2, c1, c0 = center_fit

        slope = 2*c2*y_eval + c1
        steer_rad = np.arctan(slope)
        steer_deg = np.degrees(steer_rad)
        steer_deg = np.clip(steer_deg, -30, 30)

        print("\n--- Steering ---")
        print(f"Steering Angle (deg): {steer_deg:.2f}")

    else:
        print("Could not detect both lanes")

    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))
    result = cv2.addWeighted(img_rgb, 1, newwarp, 0.8, 0)

    plt.figure(figsize=(16,5))
    
    plt.subplot(1,3,1)
    cv2.polylines(img_rgb, np.int32([src_points]), True, (255,0,0), 2)
    plt.imshow(img_rgb)
    plt.title("ROI")

    plt.subplot(1,3,2)
    plt.imshow(closed_warped, cmap='gray')
    plt.title("Warped Binary")

    plt.subplot(1,3,3)
    plt.imshow(color_warp)
    plt.title("Lanes + Center")

    plt.figure(figsize=(10,6))
    plt.imshow(result)
    plt.title("Final Output")

    plt.show()

process_lane_image('6.png')