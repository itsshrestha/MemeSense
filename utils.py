import cv2
import numpy as np

def draw_header(img, title="MEME POSE CAMERA", pose_name=None, confidence=0.0):
    """
    Draws a modern top header HUD on the frame.
    """
    h, w = img.shape[:2]
    header_h = 50
    
    # Header background overlay (semi-transparent dark)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    
    # Bottom accent line
    line_color = (0, 220, 255) if pose_name else (100, 100, 100)
    cv2.line(img, (0, header_h), (w, header_h), line_color, 2)

    # Title text
    cv2.putText(img, title, (15, 32), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Status / Detected Pose badge
    if pose_name:
        status_str = f"ACTIVE POSE: {pose_name.upper()} ({int(confidence * 100)}%)"
        cv2.putText(img, status_str, (w - 380, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "STATUS: SEARCHING FOR POSE...", (w - 340, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    return img

def create_side_by_side_view(camera_frame, meme_card, target_height=480):
    """
    Combines live camera frame and meme card side-by-side into a single window.
    """
    # Resize camera frame to match target height maintaining aspect ratio
    ch, cw = camera_frame.shape[:2]
    cam_scale = target_height / ch
    cam_w = int(cw * cam_scale)
    cam_resized = cv2.resize(camera_frame, (cam_w, target_height), interpolation=cv2.INTER_AREA)

    # Resize meme card to match target height and width
    mh, mw = meme_card.shape[:2]
    card_scale = target_height / mh
    card_w = int(mw * card_scale)
    card_resized = cv2.resize(meme_card, (card_w, target_height), interpolation=cv2.INTER_AREA)

    # Add divider line
    divider = np.zeros((target_height, 4, 3), dtype=np.uint8)
    divider[:] = (0, 200, 255)

    # Combine side by side
    combined = np.hstack((cam_resized, divider, card_resized))
    
    # Add bottom control bar hint
    ctrl_bar_h = 30
    total_w = combined.shape[1]
    ctrl_bar = np.zeros((ctrl_bar_h, total_w, 3), dtype=np.uint8)
    ctrl_bar[:] = (20, 20, 25)
    
    cv2.putText(ctrl_bar, "Controls: Press 'd' to toggle landmarks skeleton | Press 'q' or ESC to exit", 
                (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    final_window = np.vstack((combined, ctrl_bar))
    return final_window
