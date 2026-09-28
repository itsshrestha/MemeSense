import cv2
import time
import argparse
import sys
from pose_detector import PoseDetector
from meme_manager import MemeManager
from utils import draw_header, create_side_by_side_view

def run_app(camera_id=0, width=640, height=480, test_mode=False):
    """
    Main loop for Meme Pose Camera App with 1.5-second reaction hold timer.
    Gestures:
    - Both hands on head ➔ Stressed Monkey Meme 🐒
    - Pointing at chest ➔ 'Who, me?' Meme 🤔
    - Pointing at camera ➔ Pointing Cat Meme 🫵
    """
    print("=" * 60)
    print("🎭 Meme Pose Camera App")
    print("  - Place BOTH hands on head ➔ Stressed Monkey Meme 🐒")
    print("  - Point index finger at chest ➔ 'Who, me?' Meme 🤔")
    print("  - Point index finger at camera ➔ Pointing Cat Meme 🫵")
    print("  - Reaction holds on screen for 1.5 seconds.")
    print("  - Press 'd' to toggle landmark debug overlays.")
    print("  - Press 'q' or ESC to exit.")
    print("=" * 60)

    detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    meme_mgr = MemeManager()
    
    show_landmarks = False
    
    # 1.5-second hold state variables
    HOLD_DURATION = 1.5  # seconds to hold reaction image on screen
    reaction_until = 0.0
    active_pose_key = None
    active_confidence = 0.0

    if test_mode:
        print("\n[INFO] Running in Test Mode...")
        test_img = cv2.imread("assets/pointing_at_others.png")
        if test_img is None:
            print("[ERROR] Asset assets/pointing_at_others.png not found!")
            return
        
        frame = cv2.resize(test_img, (width, height))
        frame = detector.find_pose(frame, draw=False)
        hands_data = detector.get_landmarks(frame)
        pose_key, confidence, desc = detector.classify_pose(hands_data, frame=frame, draw_highlights=False)
        
        print(f"[TEST RESULT] Pose Key: {pose_key} | Confidence: {confidence:.2f} | Info: {desc}")
        meme_card = meme_mgr.render_meme_card(pose_key, width, height)
        draw_header(frame, title="MEME POSE CAMERA (TEST DEMO)", pose_name=pose_key, confidence=confidence)
        window = create_side_by_side_view(frame, meme_card, target_height=height)
        
        output_file = "assets/test_output_demo.png"
        cv2.imwrite(output_file, window)
        print(f"[SUCCESS] Demo output preview saved to {output_file}")
        return

    # Open Webcam Stream
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[WARNING] Could not open webcam device index {camera_id}.")
        print("[INFO] Fallback: Generating test demonstration preview image...")
        run_app(test_mode=True)
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Create Resizable Window for Full Screen Support
    window_name = "Meme Pose Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("[WARNING] Empty frame received from webcam. Exiting loop...")
                break

            current_time = time.time()

            # Mirror video for natural interaction
            frame = cv2.flip(frame, 1)

            # Detect Hand Landmarks
            frame = detector.find_pose(frame, draw=show_landmarks)
            hands_data = detector.get_landmarks(frame)

            # Classify Pose against registered gestures
            detected_pose, confidence, desc = detector.classify_pose(
                hands_data, 
                frame=frame, 
                draw_highlights=show_landmarks
            )

            # If ANY registered pose key was detected
            if detected_pose is not None:
                reaction_until = current_time + HOLD_DURATION
                active_pose_key = detected_pose
                active_confidence = confidence

            # Hold reaction image on screen for 1.5 seconds
            if current_time < reaction_until:
                display_pose_key = active_pose_key
                display_confidence = active_confidence
            else:
                display_pose_key = None
                display_confidence = 0.0

            # Draw HUD Header & Render Meme Card
            draw_header(frame, title="MEME POSE CAMERA", pose_name=display_pose_key, confidence=display_confidence)
            meme_card = meme_mgr.render_meme_card(display_pose_key, target_width=width, target_height=height)

            # Create side-by-side split screen view
            composite_window = create_side_by_side_view(frame, meme_card, target_height=height)

            # Display output window (Resizable/Full-screen support)
            cv2.imshow(window_name, composite_window)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print("[INFO] Quitting application...")
                break
            elif key in (ord('d'), ord('D')):
                show_landmarks = not show_landmarks
                status_str = "ON" if show_landmarks else "OFF"
                print(f"[INFO] Toggled landmark debug overlays: {status_str}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meme Pose Camera Application")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Webcam frame width")
    parser.add_argument("--height", type=int, default=480, help="Webcam frame height")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()

    run_app(camera_id=args.camera, width=args.width, height=args.height, test_mode=args.test)
