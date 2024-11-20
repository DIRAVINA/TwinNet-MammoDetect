import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp 
import joblib
import numpy as np
from skimage.feature import hog
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

knn_model = joblib.load('/Users/noorfathima/Documents/college/year 3/sem 5/Machine Learning/ML package/researchpapermodel_SLRwtKNNandSVM/knn_model.pkl') 
svm_model = joblib.load('/Users/noorfathima/Documents/college/year 3/sem 5/Machine Learning/ML package/researchpapermodel_SLRwtKNNandSVM/svm_model.pkl')
scaler = joblib.load('/Users/noorfathima/Documents/college/year 3/sem 5/Machine Learning/ML package/researchpapermodel_SLRwtKNNandSVM/scaler.pkl')  

hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True
}

def extract_hog_features(image):
    image = cv2.resize(image, (64, 64))  
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_image, **hog_params)
    return hog_features

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
'''
prediction_display_time = 2  # seconds
pred_time = None
knn_pred, svm_pred = None, None
'''
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(framergb)
    hand_landmarks = results.multi_hand_landmarks
    
    if hand_landmarks:
        for handLMs in hand_landmarks:
            
            mp_drawing.draw_landmarks(frame, handLMs, mp_hands.HAND_CONNECTIONS)

            h, w, c = frame.shape
            x_max, y_max, x_min, y_min = 0, 0, w, h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max = max(x, x_max)
                x_min = min(x, x_min)
                y_max = max(y, y_max)
                y_min = min(y, y_min)

            x_min = max(0, x_min - 20)
            x_max = min(w, x_max + 20)
            y_min = max(0, y_min - 20)
            y_max = min(h, y_max + 20)

            roi = frame[y_min:y_max, x_min:x_max]

            # Check if ROI is valid
            if roi.size > 0: 
                hog_features = extract_hog_features(roi)

                combined_features = hog_features.reshape(1, -1)  # Reshape for the scaler

                if combined_features.shape[1] == 1764: 
                    scaled_features = scaler.transform(combined_features)

                    knn_pred = knn_model.predict(scaled_features)[0]
                    svm_pred = svm_model.predict(scaled_features)[0]
                    #if pred_time is not None and (time.time() - pred_time) < prediction_display_time:
                    cv2.putText(frame, f"KNN: {knn_pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(frame, f"SVM: {svm_pred}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    #time.sleep(2)
    
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break loop
        break

cap.release()
cv2.destroyAllWindows()
