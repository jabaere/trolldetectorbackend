from deepface import DeepFace
import cv2
import numpy as np

# Dummy image (required to trigger model loading)
dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

# This will preload the models: age, gender, emotion, race
DeepFace.analyze(img_path=dummy_img, actions=["age", "gender", "emotion", "race"], enforce_detection=False)


print("Models preloaded!")