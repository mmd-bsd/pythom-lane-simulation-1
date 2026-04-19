import cv2
import time

def test_camera_fps(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Request maximum possible frame rate
    cap.set(cv2.CAP_PROP_FPS, 120)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    print(f"Testing Camera {camera_index} Max FPS...")
    print("Press Ctrl+C in the terminal or ESC in the video window to stop.")

    # Check what resolution and FPS the camera actually agreed to provide
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera negotiated: {actual_w}x{actual_h} at target {actual_fps} FPS")

    prev_time = time.time()
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Camera might have disconnected.")
            break
        
        current_time = time.time()
        
        # Calculate average FPS every 1 second to make it readable in the terminal
        frame_count += 1
        elapsed = current_time - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"Camera FPS: {fps:.2f}")
            start_time = current_time
            frame_count = 0

        # We show the frame just so you know the camera is actively working
        cv2.imshow("Camera Test - Press ESC to exit", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == "__main__":
    test_camera_fps(2)  # Change 0 to 1 or 2 if the wrong camera opens