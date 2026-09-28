import sys

print("Python version:", sys.version)

try:
    import mediapipe as mp
    print("MediaPipe imported successfully. Version:", getattr(mp, '__version__', 'unknown'))
    print("MediaPipe dir attributes:", [a for a in dir(mp) if not a.startswith('_')])
except Exception as e:
    print("Error importing mediapipe:", e)

print("\n--- Testing Solutions Imports ---")
try:
    import mediapipe.solutions.hands as mp_hands
    print("✅ Successfully imported: mediapipe.solutions.hands")
except Exception as e:
    print("❌ Failed mediapipe.solutions.hands:", e)

try:
    import mediapipe.python.solutions.hands as mp_hands
    print("✅ Successfully imported: mediapipe.python.solutions.hands")
except Exception as e:
    print("❌ Failed mediapipe.python.solutions.hands:", e)

try:
    from mediapipe.tasks.python.vision import HandLandmarker
    print("✅ Successfully imported: mediapipe.tasks.python.vision.HandLandmarker")
except Exception as e:
    print("❌ Failed mediapipe.tasks.python.vision.HandLandmarker:", e)
