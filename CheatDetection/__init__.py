import cv2
import numpy as np
from xgboost import XGBClassifier
import mediapipe as mp
import os

class CheatDetection:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        self.model = XGBClassifier()
        xgboost_model_path = os.path.join(
            "/Users/luqman/Documents/Cheating-Detection/CheatDetection", "XGB_BiCD_Tuned_GPU_02.model"
        )

        self.model.load_model(xgboost_model_path)
        self.model.set_params(**{"predictor": "gpu_predictor"})

    def GeneratePose(self, img):
        return img

    def DetectCheat(self, ShowPose=True, img=None):
        cheating = False
        if ShowPose:
            OutputImage = img.copy()
        else:
            OutputImage = None
        
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks is not None:
            landmarks = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]
            
            features = self.extract_features(landmarks)
            if features is not None:
                prediction = self.model.predict([features])
                
                if prediction:
                    cheating = True
                    OutputImage = self.draw_bounding_rectangle(OutputImage, landmarks)

        return OutputImage, cheating

    def extract_features(self, landmarks):
        if len(landmarks) == 0:
            return None
        
        features = []

        for i in range(len(landmarks)):
            for j in range(i + 1, len(landmarks)):
                x1, y1 = landmarks[i]
                x2, y2 = landmarks[j]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                features.append(distance)

        while len(features) > 75:
            features.pop()

        while len(features) < 75:
            features.append(0.0)

        return features



    def draw_bounding_rectangle(self, image, landmarks):
        image_height, image_width, _ = image.shape
        min_x = min(landmarks[:, 0])
        min_y = min(landmarks[:, 1])
        max_x = max(landmarks[:, 0])
        max_y = max(landmarks[:, 1])
        cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
        return image

cd = CheatDetection()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, prediction = cd.DetectCheat(img=frame)

    cv2.putText(frame, f"Prediction: {'Cheat Detected!' if prediction else 'No Cheat Detected'}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    if prediction:
        cv2.imwrite("cheating_screenshot.png", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

