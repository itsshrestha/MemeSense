import os
import cv2
import urllib.request
import numpy as np

class HandDetector:
    """
    Real-time Hand Detector using MediaPipe Tasks API (HandLandmarker).
    Detects gestures:
    - 'absolute_cinema': Both hands raised upward (~90 deg) with fingers open wide (Absolute Cinema).
    - 'hands_on_head': Both hands placed on top of head.
    - 'hands_clamped_mouth': Both hands clasped/clamped together over nose & mouth (Anime anxious/domain pose).
    - 'reading_paper': Holding a paper/sheet with both hands in front of chest (Jimmy Butler).
    - 'pointing_at_self': Pointing index finger at chest ("Who, me?").
    - 'pointing_at_others': Pointing index finger forward / at camera ("YOU!").
    - 'pointing_sideways': Pointing thumb or index finger sideways ("Look at this guy" - Jerry).
    """
    FINGERTIP_IDS = [4, 8, 12, 16, 20]
    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = "hand_landmarker.task"

    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.max_num_hands = max_num_hands
        self.landmarker = None
        self.mp = None
        self.results = None

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self.mp = mp

            if not os.path.exists(self.MODEL_PATH):
                print(f"[HandDetector] Downloading MediaPipe Hand Landmarker model ({self.MODEL_PATH})...")
                try:
                    urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
                    print("[HandDetector] Download completed successfully!")
                except Exception as dl_err:
                    print(f"[HandDetector] Could not download model online: {dl_err}")

            if os.path.exists(self.MODEL_PATH):
                base_options = mp_python.BaseOptions(model_asset_path=self.MODEL_PATH)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.landmarker = vision.HandLandmarker.create_from_options(options)
                print("[HandDetector] Successfully initialized MediaPipe Tasks HandLandmarker! 🖐️")
        except Exception as e:
            print(f"[HandDetector] Initialization error: {e}")

    def find_hands(self, img, draw=False):
        """
        Processes frame with HandLandmarker and draws hand skeleton connections if draw is True.
        """
        self.results = None
        if self.landmarker:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=img_rgb)
            self.results = self.landmarker.detect(mp_image)
            
            if self.results and self.results.hand_landmarks and draw:
                h, w, _ = img.shape
                connections = [
                    (0,1), (1,2), (2,3), (3,4),
                    (0,5), (5,6), (6,7), (7,8),
                    (5,9), (9,10), (10,11), (11,12),
                    (9,13), (13,14), (14,15), (15,16),
                    (13,17), (17,18), (18,19), (19,20),
                    (0,17)
                ]
                for hand_lms in self.results.hand_landmarks:
                    for p1, p2 in connections:
                        pt1 = (int(hand_lms[p1].x * w), int(hand_lms[p1].y * h))
                        pt2 = (int(hand_lms[p2].x * w), int(hand_lms[p2].y * h))
                        cv2.line(img, pt1, pt2, (0, 255, 200), 2, cv2.LINE_AA)
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        return img

    def get_hand_landmarks(self, img):
        """
        Returns list of hands with 21 landmark dicts containing (cx, cy) pixel coordinates.
        """
        all_hands = []
        h, w, c = img.shape

        if self.landmarker and self.results and self.results.hand_landmarks:
            for hand_idx, hand_lms in enumerate(self.results.hand_landmarks):
                landmarks = []
                for id, lm in enumerate(hand_lms):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append({
                        'id': id,
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'cx': cx,
                        'cy': cy
                    })
                
                label = "Hand"
                if self.results.handedness and hand_idx < len(self.results.handedness):
                    label = self.results.handedness[hand_idx][0].category_name

                all_hands.append({
                    'label': label,
                    'landmarks': landmarks
                })
        return all_hands

    def highlight_fingertips(self, img, hands_data):
        """
        Highlights fingertip landmarks with glowing circles and labels.
        """
        total_open_fingers = 0
        h, w, _ = img.shape

        for hand in hands_data:
            lms = hand['landmarks']
            if len(lms) < 21:
                continue

            open_fingers = []
            if abs(lms[4]['cx'] - lms[2]['cx']) > (w * 0.035):
                open_fingers.append(0)

            if lms[8]['cy'] < lms[6]['cy']:
                open_fingers.append(1)
            if lms[12]['cy'] < lms[10]['cy']:
                open_fingers.append(2)
            if lms[16]['cy'] < lms[14]['cy']:
                open_fingers.append(3)
            if lms[20]['cy'] < lms[18]['cy']:
                open_fingers.append(4)

            total_open_fingers += len(open_fingers)

            colors = [(0, 255, 255), (0, 255, 0), (255, 0, 255), (0, 165, 255), (255, 255, 0)]
            
            for i, tip_id in enumerate(self.FINGERTIP_IDS):
                tip = lms[tip_id]
                cx, cy = tip['cx'], tip['cy']
                color = colors[i % len(colors)]
                
                is_open = (i in open_fingers)
                radius = 12 if is_open else 6
                thickness = -1 if is_open else 2

                cv2.circle(img, (cx, cy), radius + 5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), radius, color, thickness, cv2.LINE_AA)
                
                label_str = f"{self.FINGER_NAMES[i]}"
                cv2.putText(img, label_str, (cx - 18, max(20, cy - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return total_open_fingers

    def _count_open_fingers(self, lms, w):
        open_count = 0
        if abs(lms[4]['cx'] - lms[2]['cx']) > (w * 0.03): open_count += 1
        if lms[8]['cy'] < lms[6]['cy']:   open_count += 1
        if lms[12]['cy'] < lms[10]['cy']: open_count += 1
        if lms[16]['cy'] < lms[14]['cy']: open_count += 1
        if lms[20]['cy'] < lms[18]['cy']: open_count += 1
        return open_count

    def is_absolute_cinema(self, hands_data, img_shape):
        """
        Detects both hands raised upward (~90 deg) with fingers open wide (Absolute Cinema meme).
        """
        if not hands_data or len(hands_data) < 2:
            return False, 0.0

        h, w, _ = img_shape
        h1_lms = hands_data[0]['landmarks']
        h2_lms = hands_data[1]['landmarks']

        w1_y = h1_lms[0]['cy']
        w2_y = h2_lms[0]['cy']

        if w1_y > 0.55 * h or w2_y > 0.55 * h:
            return False, 0.0

        w1_x = h1_lms[0]['cx']
        w2_x = h2_lms[0]['cx']
        wrist_span = abs(w1_x - w2_x)

        hands_spread_out = wrist_span > (0.35 * w)

        open_h1 = self._count_open_fingers(h1_lms, w)
        open_h2 = self._count_open_fingers(h2_lms, w)

        if hands_spread_out and open_h1 >= 3 and open_h2 >= 3:
            return True, 0.98

        return False, 0.0

    def is_hands_on_head(self, hands_data, img_shape):
        """
        Detects hands on head: REQUIRES BOTH HANDS placed on top of head.
        Hands must also be SPREAD APART horizontally (not clamped together),
        to avoid false triggers when user is doing the clamped-mouth pose.
        """
        if not hands_data or len(hands_data) < 2:
            return False, 0.0

        h, w, _ = img_shape
        head_top_y_limit = h * 0.45

        h1_lms = hands_data[0]['landmarks']
        h2_lms = hands_data[1]['landmarks']

        # Reject if wrists are too close together (clamped = hands_clamped_mouth pose)
        wrist_dx = abs(h1_lms[0]['cx'] - h2_lms[0]['cx'])
        if wrist_dx < 0.22 * w:
            return False, 0.0

        hands_near_head = 0
        for hand in hands_data:
            lms = hand['landmarks']
            wrist_y = lms[0]['cy']
            index_tip_y = lms[8]['cy']
            middle_tip_y = lms[12]['cy']
            pinky_tip_y = lms[20]['cy']

            if (wrist_y < head_top_y_limit or index_tip_y < head_top_y_limit or
                    middle_tip_y < head_top_y_limit or pinky_tip_y < head_top_y_limit):
                hands_near_head += 1

        if hands_near_head >= 2:
            return True, 0.95

        return False, 0.0

    def is_hands_clamped_mouth(self, hands_data, img_shape):
        """
        Detects both hands clasped/clamped together covering the nose and mouth.
        Wrists must be clearly in nose/mouth zone (below 55% of frame height),
        well separated from is_hands_on_head which triggers above 45% height.
        """
        if not hands_data or len(hands_data) < 2:
            return False, 0.0

        h, w, _ = img_shape

        h1_lms = hands_data[0]['landmarks']
        h2_lms = hands_data[1]['landmarks']

        w1 = h1_lms[0]
        w2 = h2_lms[0]
        i1 = h1_lms[8]   # index tip hand 1
        i2 = h2_lms[8]   # index tip hand 2

        # Wrists must sit clearly in the nose/mouth region — below 55% of frame.
        # This prevents overlap with hands_on_head which requires wrists above 45%.
        wrists_in_mouth_zone = (0.50 * h <= w1['cy'] <= 0.88 * h) and (0.50 * h <= w2['cy'] <= 0.88 * h)

        # Fingertips must be pointing upward into face (30%–68% height)
        tips_in_face = (0.30 * h <= i1['cy'] <= 0.68 * h) and (0.30 * h <= i2['cy'] <= 0.68 * h)

        # Both hands must be close together horizontally (clasped)
        wrist_dx = abs(w1['cx'] - w2['cx'])
        index_dx = abs(i1['cx'] - i2['cx'])
        is_clamped = (wrist_dx < 0.25 * w) or (index_dx < 0.18 * w)

        # Fingertips should be above the wrists (hands pointing upward)
        tips_upward = (i1['cy'] < w1['cy']) and (i2['cy'] < w2['cy'])

        if wrists_in_mouth_zone and tips_in_face and is_clamped and tips_upward:
            return True, 0.98

        return False, 0.0

    def is_reading_paper(self, hands_data, img_shape):
        """
        Detects holding a paper/sheet with both hands in front of chest (Jimmy Butler meme).
        """
        if not hands_data or len(hands_data) < 2:
            return False, 0.0

        h, w, _ = img_shape
        
        w1 = hands_data[0]['landmarks'][0]
        w2 = hands_data[1]['landmarks'][0]

        in_chest_height = (0.22 * h <= w1['cy'] <= 0.85 * h) and (0.22 * h <= w2['cy'] <= 0.85 * h)
        wrist_dist_x = abs(w1['cx'] - w2['cx'])
        wrist_dist_y = abs(w1['cy'] - w2['cy'])
        
        is_paper_width = (0.12 * w <= wrist_dist_x <= 0.65 * w) and (wrist_dist_y < 0.25 * h)
        not_on_head = (w1['cy'] > 0.40 * h) and (w2['cy'] > 0.40 * h)

        if in_chest_height and is_paper_width and not_on_head:
            return True, 0.94

        return False, 0.0

    def is_pointing_at_self(self, hands_data, img_shape):
        """
        Detects hand pointing index finger at self / chest ("Who, me?").
        """
        if not hands_data:
            return False, 0.0

        h, w, _ = img_shape

        for hand in hands_data:
            lms = hand['landmarks']
            wrist = lms[0]
            index_tip = lms[8]
            middle_tip = lms[12]
            ring_tip = lms[16]

            if 0.35 * h <= wrist['cy'] <= 0.90 * h:
                index_dist = np.hypot(index_tip['cx'] - wrist['cx'], index_tip['cy'] - wrist['cy'])
                middle_dist = np.hypot(middle_tip['cx'] - wrist['cx'], middle_tip['cy'] - wrist['cy'])
                ring_dist = np.hypot(ring_tip['cx'] - wrist['cx'], ring_tip['cy'] - wrist['cy'])

                is_index_extended = (index_dist > middle_dist * 1.1) or (index_dist > ring_dist * 1.1)
                is_near_chest = (0.28 * w <= index_tip['cx'] <= 0.72 * w) and (0.38 * h <= index_tip['cy'] <= 0.85 * h)
                is_not_pointing_forward = index_tip['z'] >= wrist['z'] - 0.02

                if is_index_extended and is_near_chest and is_not_pointing_forward:
                    return True, 0.92

        return False, 0.0

    def is_pointing_at_others(self, hands_data, img_shape):
        """
        Detects pointing index finger forward at camera/others ("YOU!").
        Explicitly rejects the gesture if the middle finger is more extended
        than the index finger (= middle finger salute, not a point).
        """
        if not hands_data:
            return False, 0.0

        h, w, _ = img_shape

        for hand in hands_data:
            lms = hand['landmarks']
            wrist = lms[0]
            index_tip  = lms[8]
            middle_tip = lms[12]
            ring_tip   = lms[16]

            index_dist  = np.hypot(index_tip['cx']  - wrist['cx'], index_tip['cy']  - wrist['cy'])
            middle_dist = np.hypot(middle_tip['cx'] - wrist['cx'], middle_tip['cy'] - wrist['cy'])
            ring_dist   = np.hypot(ring_tip['cx']   - wrist['cx'], ring_tip['cy']   - wrist['cy'])

            # Hard block: if middle finger is as long or longer than index → it's a middle finger
            if middle_dist >= index_dist * 0.92:
                continue

            is_index_extended = (index_dist > middle_dist * 1.15) or (index_dist > ring_dist * 1.15)
            pointing_forward  = index_tip['z'] < (wrist['z'] - 0.035)

            if is_index_extended and pointing_forward:
                return True, 0.95

        return False, 0.0

    def is_pointing_sideways(self, hands_data, img_shape):
        """
        Detects thumb or index finger pointing sideways ("Look at that guy!" - Jerry meme).
        """
        if not hands_data:
            return False, 0.0

        h, w, _ = img_shape

        for hand in hands_data:
            lms = hand['landmarks']
            wrist = lms[0]
            thumb_tip = lms[4]
            index_tip = lms[8]

            thumb_dx = thumb_tip['cx'] - wrist['cx']
            index_dx = index_tip['cx'] - wrist['cx']

            pointing_left = (thumb_tip['cx'] < 0.25 * w) or (index_tip['cx'] < 0.25 * w) or (thumb_dx < -w * 0.14)
            pointing_right = (thumb_tip['cx'] > 0.75 * w) or (index_tip['cx'] > 0.75 * w) or (thumb_dx > w * 0.14)

            if pointing_left or pointing_right:
                return True, 0.95

        return False, 0.0

    def is_middle_finger(self, hands_data, img_shape):
        """
        Detects middle finger salute gesture:
        - Middle finger (landmark 12) extended upward (tip above its PIP joint 10)
        - Index  (8) / Ring (16) / Pinky (20) fingers curled down.
        Works on either hand.
        """
        if not hands_data:
            return False, 0.0

        for hand in hands_data:
            lms = hand['landmarks']

            middle_tip = lms[12]['cy']
            middle_pip = lms[10]['cy']

            index_tip  = lms[8]['cy']
            index_pip  = lms[6]['cy']

            ring_tip   = lms[16]['cy']
            ring_pip   = lms[14]['cy']

            pinky_tip  = lms[20]['cy']
            pinky_pip  = lms[18]['cy']

            # Middle finger clearly up (tip well above its PIP = smaller cy)
            middle_extended = middle_tip < middle_pip - 8

            # Other three fingers curled (tip below PIP = larger cy)
            index_curled = index_tip  > index_pip
            ring_curled  = ring_tip   > ring_pip
            pinky_curled = pinky_tip  > pinky_pip

            if middle_extended and index_curled and ring_curled and pinky_curled:
                return True, 0.97

        return False, 0.0
