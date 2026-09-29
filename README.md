# 🎭 MemeSense

**MemeSense** is a real-time hand gesture-to-meme reaction system. It uses your webcam and AI-powered hand landmark detection to recognize poses and instantly display matching meme reactions on screen — no buttons, no keyboard, just vibes.

---

## 📸 How It Works

The app splits your screen into two panels:

| Left Panel | Right Panel |
|---|---|
| Live webcam feed with hand skeleton overlay | Meme reaction card matching your pose |

When no gesture is detected, the right panel shows a **"MAKE A POSE"** guide listing all available gestures. The moment a pose is matched, the meme card holds on screen for **1.5 seconds** before rescanning.

---

## 🎮 Supported Gestures & Memes

| # | Gesture | How to Do It | Meme Reaction |
|---|---|---|---|
| 1 | **Absolute Cinema** | Both hands raised ~90° upward, all fingers open wide | 🎬 Absolute Cinema |
| 2 | **Stressed Monkey** | Both hands spread on top of head (hands apart) | 🐒 Stressed Monkey |
| 3 | **Anxious Domain** | Both hands clasped together covering nose & mouth, wrists at mouth level | 🙏😨 Anxious Domain Expansion |
| 4 | **Reading Paper** | Both hands in front of chest holding an imaginary sheet | 📄😁 Jimmy Butler Reading |
| 5 | **Who, Me?** | One index finger pointing at your own chest | 🤔 Who, Me? |
| 6 | **Gorilla Says No** | Middle finger extended, all other fingers curled | 🦍🖕 Gorilla Flipping the Bird |
| 7 | **Pointing at You** | Index finger pointing forward at the camera | 🫵😹 Pointing Cat |
| 8 | **Look at This Guy** | Thumb or index finger pointing far to the side | 😂👍 Laughing Jerry |

> **Tips:**
> - Gestures 1–4 require **both hands** visible in frame.
> - Gestures 5–8 work with **one hand**.
> - Keep your hands clearly visible and well-lit for best detection accuracy.

---

## 🗂️ Project Structure

```
Meme/
├── main.py                  # Entry point — main loop, camera capture, UI rendering
├── hand_detector.py         # MediaPipe hand landmark detection + all gesture logic
├── pose_detector.py         # Gesture classifier — maps landmark data to meme keys
├── meme_manager.py          # Loads meme images and renders the reaction card panel
├── utils.py                 # Drawing helpers (skeleton overlay, FPS, etc.)
├── debug_mediapipe.py       # Standalone script to verify MediaPipe installation
├── hand_landmarker.task     # MediaPipe hand landmark model file (bundled)
├── requirements.txt         # Python package dependencies
└── assets/                  # Meme reaction images
    ├── absolute_cinema.png
    ├── hands_on_head.png
    ├── hands_clamped_mouth.png
    ├── reading_paper.png
    ├── pointing_at_self.png
    ├── pointing_at_others.png
    ├── pointing_sideways.png
    └── middle_finger.png
```

---

## ⚙️ Requirements

- **Python** 3.10 – 3.12
- **Webcam** (built-in or USB)
- **OS**: Linux, Windows, or macOS

### Python Packages

```
opencv-python
mediapipe>=0.10.35
pillow
numpy
```

> ⚠️ **Important:** This project uses `mediapipe>=0.10.35` which does **not** expose `mediapipe.solutions.hands`.
> It uses the newer `mediapipe.tasks.python.vision.HandLandmarker` API instead. Do not downgrade MediaPipe.

---

## 🚀 Setup & Running

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Meme
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify MediaPipe installation (optional)

```bash
python3 debug_mediapipe.py
```

You should see:
```
✅ Successfully imported: mediapipe.tasks.python.vision.HandLandmarker
```

### 5. Run the app

```bash
python3 main.py
```

---

## 🖥️ Controls

| Key | Action |
|---|---|
| `H` | Toggle fingertip highlight dots on/off |
| `Q` | Quit the application |

The window is **resizable** — drag or maximize it to full screen.

---

## 🛠️ Troubleshooting

### Webcam not opening
- Make sure no other application is using the camera.
- Try changing the camera index in `main.py`: `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)`.

### `hand_landmarker.task` not found
- The model file must be in the project root directory alongside `main.py`.
- Download from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models if missing.

### Gestures not triggering / wrong meme firing
- Ensure your hands are clearly visible and well-lit.
- Keep both hands within the left camera panel.
- For two-hand gestures, both hands must be fully in frame.
- Low light degrades landmark accuracy significantly.

### `mediapipe.solutions` import error
- This is expected on MediaPipe >= 0.10. The project already uses the Tasks API.
- Run `debug_mediapipe.py` to confirm your setup is correct.

---

## 🧠 Architecture Overview

```
Webcam Frame
     |
     v
HandDetector (hand_detector.py)
  └─ MediaPipe HandLandmarker → 21 landmarks per hand
     |
     v
PoseDetector (pose_detector.py)
  └─ Evaluates gesture rules in priority order:
     1. Absolute Cinema
     2. Hands Clamped Mouth  ← checked before Hands on Head
     3. Hands on Head
     4. Reading Paper
     5. Pointing at Self
     6. Middle Finger         ← checked before Pointing at Others
     7. Pointing at Others
     8. Pointing Sideways
     |
     v
MemeManager (meme_manager.py)
  └─ Loads matched meme image → renders reaction card
     |
     v
main.py  →  Split-screen display (camera | reaction card)
```

---

## 📦 Adding New Gestures

1. **Add detection logic** in `hand_detector.py` — create a new `is_<gesture_name>` method.
2. **Register in classifier** in `pose_detector.py` — add a check inside `classify_pose()` in priority order.
3. **Register the meme** in `meme_manager.py` — call `self.register_meme(key, image_file, title)` in `_register_default_memes()`.
4. **Add the image** to `assets/<gesture_name>.png`.
5. **Update the hints list** in `meme_manager.py` → `render_meme_card()` idle panel.

---

## 📄 License

This project is for fun and educational purposes. Meme images belong to their respective creators/owners.
