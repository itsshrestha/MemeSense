import os
import cv2
import numpy as np

class MemeManager:
    def __init__(self, assets_dir="assets"):
        self.assets_dir = assets_dir
        self.memes = {}
        self._register_default_memes()

    def _register_default_memes(self):
        self.register_meme("absolute_cinema",    "absolute_cinema.png",    "ABSOLUTE CINEMA")
        self.register_meme("hands_on_head",      "hands_on_head.png",      "OH NO! STRESSED MONKEY")
        self.register_meme("hands_clamped_mouth","hands_clamped_mouth.png","ANXIOUS DOMAIN EXPANSION")
        self.register_meme("reading_paper",      "reading_paper.png",      "READING THE STAT SHEET")
        self.register_meme("pointing_at_self",   "pointing_at_self.png",   "WHO, ME?")
        self.register_meme("pointing_at_others", "pointing_at_others.png", "YOU! POINTING AT YOU")
        self.register_meme("pointing_sideways",  "pointing_sideways.png",  "LOOK AT THIS GUY!")
        self.register_meme("middle_finger",      "middle_finger.png",      "GORILLA SAYS NO!")

    def register_meme(self, pose_key, image_filename, title, description=""):
        filepath = os.path.join(self.assets_dir, image_filename)
        img = None
        if os.path.exists(filepath):
            img = cv2.imread(filepath)
        self.memes[pose_key] = {
            'filepath': filepath,
            'title': title,
            'description': description,
            'image': img
        }

    def get_meme(self, pose_key):
        return self.memes.get(pose_key, None)

    def render_meme_card(self, pose_key, target_width, target_height):
        card = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        card[:] = (20, 20, 24)

        banner_h = 60
        meme_info = self.get_meme(pose_key)

        if pose_key and meme_info and meme_info['image'] is not None:
            raw_img = meme_info['image']
            img_area_h = target_height - banner_h
            h, w = raw_img.shape[:2]
            scale = min(target_width / w, img_area_h / h)
            nw, nh = int(w * scale), int(h * scale)
            resized_img = cv2.resize(raw_img, (nw, nh), interpolation=cv2.INTER_AREA)
            x_offset = (target_width - nw) // 2
            y_offset = banner_h + (img_area_h - nh) // 2
            card[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized_img

            cv2.rectangle(card, (0, 0), (target_width, banner_h), (35, 35, 45), -1)
            cv2.line(card, (0, banner_h), (target_width, banner_h), (0, 220, 255), 2)
            cv2.putText(card, "MATCHED REACTION", (15, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(card, meme_info['title'], (15, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(card, (0, 0), (target_width, banner_h), (30, 30, 38), -1)
            cv2.putText(card, "REACTION PANEL", (15, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
            cv2.putText(card, "Waiting for Meme Pose...", (15, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2, cv2.LINE_AA)

            text_str = "MAKE A POSE"
            font_scale, font_thickness = 1.1, 3
            text_size, _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = (target_width - text_size[0]) // 2
            text_y = target_height // 2 - 75

            cv2.putText(card, text_str, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 220, 255), font_thickness, cv2.LINE_AA)

            hints = [
                "* Hands raised 90 open  ->  Absolute Cinema",
                "* Both hands on head    ->  Stressed Monkey",
                "* Hands clasped mouth   ->  Anxious Domain",
                "* Both hands hold paper ->  Jimmy Butler",
                "* Point at chest        ->  Who, me?",
                "* Point at camera       ->  Pointing Cat",
                "* Point sideways        ->  Laughing Jerry",
                "* Middle finger up      ->  Gorilla Says No!",
            ]
            for i, hint in enumerate(hints):
                color = (0, 220, 255) if i == 0 else (160, 160, 170)
                cv2.putText(card, hint, (target_width // 2 - 185, text_y + 30 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        return card
