import cv2
import numpy as np
from hand_detector import HandDetector

class PoseDetector:
    """
    Combined Pose & Hand Landmark Detector.
    Classifies registered meme gestures:
    - 'absolute_cinema': Both hands raised ~90 deg upward with fingers open wide (Absolute Cinema).
    - 'hands_on_head': Both hands placed on top of head.
    - 'hands_clamped_mouth': Both hands clasped over nose & mouth (Anime anxious/domain pose).
    - 'reading_paper': Holding paper/sheet with both hands (Jimmy Butler reading paper).
    - 'pointing_at_self': Pointing index finger at chest ("Who, me?").
    - 'pointing_at_others': Pointing index finger at camera/others ("YOU! 🫵").
    - 'pointing_sideways': Pointing thumb/index sideways ("Look at that guy!" - Jerry).
    - 'middle_finger': Middle finger extended, others curled (Gorilla flipping the bird).
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hand_detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def find_pose(self, img, draw=False):
        """
        Finds hand landmarks, drawing skeleton connections only if draw is True.
        """
        img = self.hand_detector.find_hands(img, draw=draw)
        return img

    def get_landmarks(self, img):
        """
        Returns hand landmarks structure.
        """
        return self.hand_detector.get_hand_landmarks(img)

    def classify_pose(self, hands_data, frame=None, draw_highlights=False):
        """
        Classifies pose against registered meme gestures.
        Returns (pose_key, confidence, description).
        """
        if not hands_data or len(hands_data) == 0:
            return None, 0.0, "Waiting for pose..."

        frame_shape = frame.shape if frame is not None else (480, 640, 3)

        if draw_highlights and frame is not None:
            self.hand_detector.highlight_fingertips(frame, hands_data)

        # 1. Check ABSOLUTE CINEMA (Both hands raised 90 deg upward with open fingers)
        is_cinema, confidence = self.hand_detector.is_absolute_cinema(hands_data, frame_shape)
        if is_cinema:
            return "absolute_cinema", confidence, "ABSOLUTE CINEMA Pose (Hands Raised 90° Upward)"

        # 2. Check BOTH hands clasped over mouth/nose — checked BEFORE hands_on_head
        #    because clamped hands near face could falsely trigger the monkey pose
        is_clamped, confidence = self.hand_detector.is_hands_clamped_mouth(hands_data, frame_shape)
        if is_clamped:
            return "hands_clamped_mouth", confidence, "Anxious Clasped Hands Pose (Hands Over Mouth)"

        # 3. Check BOTH hands on head (Stressed Monkey Meme)
        #    Requires hands to be spread apart horizontally (not clamped)
        is_on_head, confidence = self.hand_detector.is_hands_on_head(hands_data, frame_shape)
        if is_on_head:
            return "hands_on_head", confidence, "Stressed Monkey Pose (Hands on Head)"

        # 4. Check BOTH hands holding paper (Jimmy Butler Meme)
        is_reading, confidence = self.hand_detector.is_reading_paper(hands_data, frame_shape)
        if is_reading:
            return "reading_paper", confidence, "Reading Paper Pose (Jimmy Butler Meme)"

        # 5. Check Pointing at Self ("Who, Me?" Meme)
        is_pointing_self, confidence = self.hand_detector.is_pointing_at_self(hands_data, frame_shape)
        if is_pointing_self:
            return "pointing_at_self", confidence, "Who, Me? Pose (Pointing at Chest)"

        # 6. Check Middle Finger — BEFORE pointing_at_others to prevent overlap
        is_mid_finger, confidence = self.hand_detector.is_middle_finger(hands_data, frame_shape)
        if is_mid_finger:
            return "middle_finger", confidence, "Middle Finger! (Gorilla Meme)"

        # 7. Check Pointing at Others / Camera ("YOU!" Cat Meme)
        is_pointing_others, confidence = self.hand_detector.is_pointing_at_others(hands_data, frame_shape)
        if is_pointing_others:
            return "pointing_at_others", confidence, "Pointing at You! Pose (Pointing Forward)"

        # 8. Check Pointing Sideways (Jerry Meme)
        is_pointing_side, confidence = self.hand_detector.is_pointing_sideways(hands_data, frame_shape)
        if is_pointing_side:
            return "pointing_sideways", confidence, "Pointing Sideways! Pose (Jerry Meme)"

        return None, 0.0, "Idle View"
