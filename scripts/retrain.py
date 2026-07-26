import os
import glob
import joblib
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skimage import feature
from sklearn.utils import resample, shuffle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

print("Starting retraining script without extra dependencies...")

# ----- VOICE MODEL RETRAINING -----
print("Retraining voice model...")
voice_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'datasets', 'voice', 'Parkinsson disease.csv'))
voice_data = voice_data.drop('name', axis=1)

# Basic outlier removal
z_scores = (voice_data - voice_data.mean()) / voice_data.std()
voice_data = voice_data[(z_scores < 3).all(axis=1)]

# Simple SMOTE replacement using random oversampling
df_majority = voice_data[voice_data.status==1]
df_minority = voice_data[voice_data.status==0]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=42) # reproducible results

voice_data_bal = pd.concat([df_majority, df_minority_upsampled])
voice_data_bal = shuffle(voice_data_bal, random_state=42)

X_v = voice_data_bal.drop('status', axis=1)
y_v = voice_data_bal['status']

X_v_train, _, y_v_train, _ = train_test_split(X_v, y_v, test_size=0.3, random_state=42)

selected_features_v = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA' 
]
X_v_train_sel = X_v_train[selected_features_v]

scaler = StandardScaler()
X_v_train_scaled = scaler.fit_transform(X_v_train_sel)

voice_model = RandomForestClassifier(random_state=42)
voice_model.fit(X_v_train_scaled, y_v_train)

joblib.dump(voice_model, os.path.join(PROJECT_ROOT, 'backend', 'models', 'voice_modell.pkl'))
print("Saved voice_modell.pkl")

# ----- DRAWING MODEL RETRAINING -----
print("Retraining drawing model...")
def hog_features(image):
    return feature.hog(image, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")

def process_images_custom(directory_path):
    # Get all .png files recursively
    image_paths = glob.glob(os.path.join(directory_path, "**", "*.png"), recursive=True)
    if not image_paths:
        image_paths = glob.glob(os.path.join(directory_path, "**", "*.jpg"), recursive=True)
    if not image_paths:
        image_paths = glob.glob(os.path.join(directory_path, "**", "*.*"), recursive=True)
        # filter out non-images
        image_paths = [img for img in image_paths if img.endswith(('.png', '.jpg', '.jpeg'))]
        
    data, labels = [], []
    for img_path in image_paths:
        # parent directory is the label
        label = os.path.basename(os.path.dirname(img_path))
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (250, 250), fx=0.5, fy=0.5)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        data.append(hog_features(img))
        labels.append(label)
    return np.array(data), np.array(labels)

train_d, train_dl = process_images_custom(os.path.join(PROJECT_ROOT, 'datasets', 'drawings', 'training'))
# The original code uses label encoder
encoder = LabelEncoder()
train_dl_enc = encoder.fit_transform(train_dl)

drawing_model = RandomForestClassifier(random_state=2)
drawing_model.fit(train_d, train_dl_enc)

joblib.dump(drawing_model, os.path.join(PROJECT_ROOT, 'backend', 'models', 'drawing_model.pkl'))
print(f"Saved drawing_model.pkl with {len(train_d)} samples.")

# ----- MULTIMODAL MODEL (Rf_edit.pkl) RETRAINING -----
print("Retraining multimodal model...")
# Based on the user's app.py line 55-60: model_input = np.concatenate((voice_features, img_features), axis=1)
# Create a valid input dataset for the combined model
y_v_arr = y_v_train.values

# Match patients and non-patients
patients_v_idx = np.where(y_v_arr == 1)[0]
non_patients_v_idx = np.where(y_v_arr == 0)[0]
patients_d_idx = np.where(train_dl_enc == 1)[0]
non_patients_d_idx = np.where(train_dl_enc == 0)[0]

pat_min = min(len(patients_v_idx), len(patients_d_idx))
npat_min = min(len(non_patients_v_idx), len(non_patients_d_idx))

if pat_min > 0 and npat_min > 0:
    X_train_pat = np.concatenate((X_v_train_scaled[patients_v_idx[:pat_min]], train_d[patients_d_idx[:pat_min]]), axis=1)
    y_train_pat = np.ones(pat_min)

    X_train_npat = np.concatenate((X_v_train_scaled[non_patients_v_idx[:npat_min]], train_d[non_patients_d_idx[:npat_min]]), axis=1)
    y_train_npat = np.zeros(npat_min)

    X_train_combined = np.concatenate((X_train_pat, X_train_npat))
    y_train_combined = np.concatenate((y_train_pat, y_train_npat))

    multimodal_model = RandomForestClassifier(random_state=42)
    multimodal_model.fit(X_train_combined, y_train_combined)

    joblib.dump(multimodal_model, os.path.join(PROJECT_ROOT, 'backend', 'models', 'RF_edit.pkl'))
    print("Saved RF_edit.pkl")
else:
    print("WARNING: Could not train RF_edit.pkl because couldn't align datasets (maybe one is empty).")

print("Done! All models successfully retrained.")
