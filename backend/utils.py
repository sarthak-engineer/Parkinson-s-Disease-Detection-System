import librosa
import cv2
import numpy as np
from skimage import feature
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def extract_voice_features(audio_file_path):
    samples, sample_rate = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=9)
    features_mean = [mfcc.mean() for mfcc in mfccs]
    voice_features = np.array(features_mean)
    return scaler.fit_transform(voice_features.reshape(-1, 1)).reshape(1, -1)

def extract_img_features(image_file_path):
    # Use imdecode to handle paths with special characters better on Windows
    with open(image_file_path, "rb") as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
    if img is None:
        raise ValueError(f"Could not read image at {image_file_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (250, 250))
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    hog_features = feature.hog(img, orientations=9,
                               pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                               transform_sqrt=True, block_norm="L1")
    return hog_features.reshape(1, -1)
