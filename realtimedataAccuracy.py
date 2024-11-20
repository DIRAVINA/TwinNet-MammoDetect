import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report

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

image_folder = "captured_images"

true_labels = []
knn_predictions = []
svm_predictions = []

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        true_label = filename.split('_')[0]
        true_labels.append(true_label)

        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        hog_features = extract_hog_features(image)
        hog_features = hog_features.reshape(1, -1) 

        scaled_features = scaler.transform(hog_features)

        knn_pred = knn_model.predict(scaled_features)[0]
        svm_pred = svm_model.predict(scaled_features)[0]
        
        knn_predictions.append(knn_pred)
        svm_predictions.append(svm_pred)

knn_accuracy = accuracy_score(true_labels, knn_predictions)
svm_accuracy = accuracy_score(true_labels, svm_predictions)

knn_class_report = classification_report(true_labels, knn_predictions)
svm_class_report = classification_report(true_labels, svm_predictions)

print(f"KNN Model Accuracy: {knn_accuracy * 100:.2f}%")
print("KNN Classification Report:")
print(knn_class_report)

print(f"SVM Model Accuracy: {svm_accuracy * 100:.2f}%")
print("SVM Classification Report:")
print(svm_class_report)
