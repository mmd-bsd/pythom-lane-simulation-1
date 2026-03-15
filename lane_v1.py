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
    
    print("Please click 4 points on the image window in this order:")
    print("Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
    
    src_points = select_roi(image_path)
    
    if src_points is None or len(src_points) != 4:
        print("Error: You must select exactly 4 points. Exiting.")
        return

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

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

    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (0, 255, 255)]
    line_index = 0

    print("\n--- Detected Polynomials (x = c2*y^2 + c1*y + c0) ---")

    for contour in contours:
        # Get bounding box for the contour to check its height
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter: Area > 150 AND Height must be at least 25% of the image height
        if cv2.contourArea(contour) > 150 and h > (height * 0.25):
            contour_mask = np.zeros_like(warped)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            line_y, line_x = np.nonzero(contour_mask)

            if len(line_x) > 100:
                fit = np.polyfit(line_y, line_x, 2)
                c2, c1, c0 = fit
                print(f"Line {line_index + 1} -> c2: {c2:.6f}, c1: {c1:.6f}, c0: {c0:.2f}")

                min_y = np.min(line_y)
                max_y = np.max(line_y)
                ploty = np.linspace(min_y, max_y, int(max_y - min_y + 1))

                fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
                pts = np.array([np.transpose(np.vstack([fitx, ploty]))])
                
                color = colors[line_index % len(colors)]
                cv2.polylines(color_warp, np.int32([pts]), isClosed=False, color=color, thickness=25)
                
                line_index += 1

    if line_index == 0:
        print("No lines detected.")
    print("--------------------------------------------------------\n")

    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height)) 
    result = cv2.addWeighted(img_rgb, 1, newwarp, 0.8, 0)

    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    cv2.polylines(img_rgb, np.int32([src_points]), isClosed=True, color=(255, 0, 0), thickness=2)
    plt.imshow(img_rgb)
    plt.title('Original + Dynamic ROI')
    
    plt.subplot(1, 3, 2)
    plt.imshow(closed_warped, cmap='gray')
    plt.title('Connected Lines')
    
    plt.subplot(1, 3, 3)
    plt.imshow(color_warp)
    plt.title('Fitted Lines (Bird-Eye)')
    
    plt.tight_layout()

    plt.figure(figsize=(12, 8))
    plt.imshow(result)
    plt.title('Final Output Overlay')
    plt.tight_layout()
    
    plt.show()

process_lane_image('1.png')