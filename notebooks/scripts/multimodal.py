#!/usr/bin/env python
# coding: utf-8

# # Parkinson Disease Early Detection
# in this project, we will use machine learning algorithms to detect parkinson disease using multimodal dataset (voice records and images).
# <b> in the begining, </b> we will install and import necessary modules and read our dataset.

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[79]:


# import modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import librosa


# ***in the begining,*** we will deal with dataset thet contain features extracted from voice records (parkinson and healthy people), we will **preprocess** it, **visualize** it, **understand** it, and **extract features** from it.

# In[112]:


# Specify the file path
file_path = "/content/drive/MyDrive/multimodal project/datasets/voice/Parkinsson disease.csv"

# Read data from the CSV file
data = pd.read_csv(file_path)


# In[113]:


# show the first five samples
data.head()


# # Data Distribution
# we will visualize data distribution, this proess help us to understand dataset deeply.
# 

# In[82]:


plt.figure(figsize=(15, 8))

# Plot the distribution of 'MDVP:Fo(Hz)'
sns.histplot(data['MDVP:Fo(Hz)'], bins=20, kde=True, color='blue')

# Set the title and labels
plt.title('Distribution of MDVP:Fo(Hz)')
plt.xlabel('MDVP:Fo(Hz)')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# In[83]:


plt.figure(figsize=(20, 10))
plt.subplots_adjust(hspace=0.5)

# Plot multiple distributions
plt.subplot(2, 3, 1)
sns.histplot(data['MDVP:Fo(Hz)'], bins=20, kde=True, color='blue')
plt.title('Distribution of MDVP:Fo(Hz)')

plt.subplot(2, 3, 2)
sns.histplot(data['MDVP:Fhi(Hz)'], bins=20, kde=True, color='green')
plt.title('Distribution of MDVP:Fhi(Hz)')

# Add more subplots for other variables as needed

# Show the plot
plt.show()


# In[84]:


plt.figure(figsize=(15, 8))

# Create a FacetGrid with 'status' as hue
g = sns.FacetGrid(data, hue='status', height=6)
g.map(sns.histplot, 'MDVP:Fo(Hz)', bins=20, kde=True, alpha=0.5)

# Set the title and labels
plt.title('Distribution of MDVP:Fo(Hz) with Status')
plt.xlabel('MDVP:Fo(Hz)')
plt.ylabel('Frequency')

# Add a legend
g.add_legend()

# Show the plot
plt.show()


# In[85]:


# Select relevant variables
selected_vars = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'D2', 'PPE', 'status']

# Create a DataFrame with selected variables
selected_data = data[selected_vars]

# Create a pairplot
plt.figure(figsize=(15, 10))
sns.pairplot(selected_data, hue='status', diag_kind='kde', markers=['o', 's'], palette={0: 'blue', 1: 'orange'})

# Set the title
plt.suptitle('Pairplot of Selected Variables with Status', y=1.02)

# Show the plot
plt.show()


# # Data Cleaning and Exploration
# we will use pandas to clean our data and Extract important information from the features we have and knowing the types of data contained in it so that we can transform the data into the appropriate form for machine learning algorithms.

# In[86]:


# show information about dataset
data.info()


# In[87]:


# show statistical description for dataset
data.describe()


# In[88]:


# detect the number of missing values in each colum
missing_values = data.isnull().sum()
missing_values


# In[89]:


data['status'].value_counts()


# In[114]:


# Drop unnecessary column
data = data.drop('name', axis=1)

# Calculate z-score for each feature
z_scores = (data - data.mean()) / data.std()

# Remove outliers using the standard threshold = 3
data = data[(z_scores < 3).all(axis=1)]

# Check that the number of samples equals 181 and the number of features equals 13
print("Number of samples after removing outliers:", data.shape[0])
print("Number of features after removing outliers:", data.shape[1])


# In[115]:


# Calculate the correlation coefficient between each pair of features
correlation_matrix = data.corr()

# Identify columns with strong correlations (more than 0.8 or less than -0.8)
strong_correlation = np.where(np.abs(correlation_matrix) > 0.8)

# Drop the column if it has a strong correlation with any other column
for col1, col2 in zip(*strong_correlation):
    if col1 != col2 and col1 < col2:
        if col1 in data.columns and col2 in data.columns:
            data = data.drop(data.columns[col2], axis=1)


# In[116]:


data.shape


# # Balancing the data
# We noticed that the data we have is unbalanced, and in order for the result that we will obtain to be fair and so that the algorithm is not biased, we will balance the data using one of the data augmentation techniques so that the number of patient samples becomes equal to the number of healthy people’s samples.

# In[117]:


# Shuffle the dataset
data = shuffle(data, random_state=42)


# In[118]:


# Separate features (X) and target variable (y)
X = data.drop('status', axis=1)
y = data['status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("x_train shape is: {}".format(X_train.shape))
print("y_train shape is: {}".format(y_train.shape))
print("X_test shape is: {}".format(X_test.shape))
print("y_test shape is: {}".format(y_test.shape))


# In[119]:


# Display class distribution before balancing for training set
print("Class distribution before balancing for training set:")
print(y_train.value_counts())


# In[120]:


# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Display class distribution after balancing for training set
print("\nClass distribution after balancing for training set:")
y_train_balanced.value_counts()


# In[121]:


y_test.value_counts()


# In[122]:


# Calculate class distribution after balancing
class_distribution = pd.Series(y_train_balanced).value_counts()

# Plot the class distribution
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.title('Distribution of Classes after Balancing')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# # feature selection
# Here we will use one of the feature selection techniques to choose a number of features that will be input to the machine learning algorithms, as we will not train the algorithm on all the features in the dataset, but rather we will choose the best of these features.

# In[123]:


# Selecting the features based on their names
selected_features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2'
]

# Filter the training dataset to include only the selected features
X_train = X_train_balanced[selected_features]

# Filter the testing dataset to include only the selected features
X_test = X_test[selected_features]


# # Scaling
# We will use scaling techniques to make the features in the same field. This step is very important and will make the features equally important in the model. the we will split our data to train and test (80% for training and 20% for testing).

# In[124]:


# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the features
X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)


# # Reduce data dimensionality
# we will reduce the number of features using PCA algorithm.

# In[125]:


# Compute the covariance matrix
cov_matrix = np.cov(X_train.T)

# Compute the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Select the top eigenvectors based on the explained variance
total_variance = sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance
explained_variance_ratio_cumsum = np.cumsum(explained_variance_ratio)

# Choose the number of components to retain based on the explained variance ratio
n_components = 9

# Transform the original dataset into the new feature space
pca = PCA(n_components=n_components)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[126]:


# Get the names of the selected features
selected_feature_indices = pca.components_

# Assuming you have a list of feature names, you can use the selected indices to get the names
selected_features = [selected_features[i] for i in range(n_components)]


# In[127]:


selected_features


# ***Second,*** we will deal with **hand writing images dataset** (parkinson and healthy) people, we will **extract features** from it and then we will merge the features with features that extracted from voice dataset.
# *note:* this approach called ***early fusion***. you can understand this approach with this image:
# 

# 
# ![image](method.png)

# In[128]:


# import modules
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage import feature
import cv2
import numpy as np
import os
import joblib


# In[129]:


# Function to extract HOG features from an image
def hog_features(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features
# Function to process images in a directory and extract features and labels
def process_images(directory_path):
    image_paths = list(paths.list_images(directory_path))
    data = []
    labels = []
    for img_path in image_paths:
        label = img_path.split(os.path.sep)[-2] # Extract label from image path
        img = cv2.imread(img_path) # Read image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        img = cv2.resize(img, (250, 250), fx=0.5, fy=0.5) # Resize image
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # Apply thresholding
        features = hog_features(img) # Extract HOG features
        data.append(features)
        labels.append(label)
    return (np.array(data), np.array(labels))

# Function to train a RandomForestClassifier model and evaluate its performance
def extract_features(data_directory):
    training_path = os.path.sep.join([data_directory, "training"])
    testing_path = os.path.sep.join([data_directory, "testing"])
    train_data, train_labels = process_images(training_path)
    test_data, test_labels = process_images(testing_path)
    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(train_labels)
    test_labels = encoder.transform(test_labels)

    return train_data, test_data, train_labels, test_labels


# In[130]:


data_directory = "/content/drive/MyDrive/multimodal project/datasets/drawings"

imgx_train, imgx_test, imy_train, imy_test = extract_features(data_directory)


# ***now,*** we will merge features for training and testing dataset.

# In[131]:


# in training dataset
# Combine features and labels for patients
X_train_patients = np.concatenate((X_train[y_train_balanced == 1], imgx_train[imy_train == 1]), axis = 1)
y_train_patients = np.ones(X_train_patients.shape[0])

# Combine features and labels for healthy
X_train_non_patients = np.concatenate((X_train[y_train_balanced == 0], imgx_train[imy_train == 0]), axis = 1)
y_train_non_patients = np.zeros(X_train_non_patients.shape[0])

# Ensure balanced dataset by taking the minimum number of samples from each class
min_samples = min(X_train_patients.shape[0], X_train_non_patients.shape[0])
X_train_combined = np.concatenate((X_train_patients[:min_samples], X_train_non_patients[:min_samples]))
y_train_combined = np.concatenate((y_train_patients[:min_samples], y_train_non_patients[:min_samples]))

# Shuffle the combined dataset
combined_indices = np.arange(X_train_combined.shape[0])
np.random.shuffle(combined_indices)
X_train_combined = X_train_combined[combined_indices]
y_train_combined = y_train_combined[combined_indices]

print("Shape of combined features:", X_train_combined.shape)
print("Shape of combined labels:", y_train_combined.shape)


# In[132]:


# in testing dataset
# Combine features and labels for patients
X_test_patients = np.concatenate((X_test[y_test == 1], imgx_test[imy_test == 1]), axis = 1)
y_test_patients = np.ones(X_test_patients.shape[0])

# Combine features and labels for non-patients
X_test_non_patients = np.concatenate((X_test[y_test == 0], imgx_test[imy_test == 0]), axis = 1)
y_test_non_patients = np.zeros(X_test_non_patients.shape[0])

# Ensure balanced dataset by taking the minimum number of samples from each class
samples = X_test_patients.shape[0] +  X_test_non_patients.shape[0]
X_test_combined = np.concatenate((X_test_patients[:samples], X_test_non_patients[:samples]))
y_test_combined = np.concatenate((y_test_patients[:samples], y_test_non_patients[:samples]))

# Shuffle the combined dataset
combined_indices = np.arange(X_test_combined.shape[0])
np.random.shuffle(combined_indices)
X_test_combined = X_test_combined[combined_indices]
y_test_combined = y_test_combined[combined_indices]

print("Shape of combined test features:", X_test_combined.shape)
print("Shape of combined test labels:", y_test_combined.shape)


# # Trainig and Evaluation
# we will use machine learning algorithms, and we will improve it with Grid search Technique, we will use Random Forest and SVM algorithmns and compare them.

# In[136]:


# function to train the model
def train_model(model, X_train, y_train):

    # Train the model
    model.fit(X_train, y_train)

# function to evaluate the model
def evaluate_model(model, X, y):

    # Predict
    y_pred = model.predict(X)

    # Get evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    print("Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Plot confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    print("Classification Report:")
    print(classification_report(y, y_pred))

# Function to save a trained model to a file
def save_model(model, model_path):
    joblib.dump(model, model_path)

# Function to load a trained model from a file
def load_model(model_path):
    return joblib.load(model_path)


# in each algorithm we will train and evaluate the model and visualize accuracy and error for test data in each it and draw confusion matrix for testing data.
# # Random Forest Algorithm
# 

# In[134]:


# Initialize the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)

# train RF model
train_model(random_forest_model, X_train_combined, y_train_combined)


# In[ ]:


# save the model
model_path = "/content/drive/MyDrive/multimodal project/models/Rf.pkl"
save_model(random_forest_model, model_path)


# In[146]:


# Load the saved model from the file
model_path = "/content/drive/MyDrive/multimodal project/models/Rf.pkl"
loaded_model = load_model(model_path)


# In[147]:


# evaluate RF mdoel on Training set
print("Evaluation RF model on Training dataset:")
evaluate_model(loaded_model, X_train_combined, y_train_combined)


# In[148]:


# evaluate RF mdoel on Testing set
print("Evaluation RF model on testing dataset:")
evaluate_model(loaded_model, X_test_combined, y_test_combined)


# ### Grid search
# we will apply Grid Search Algorithm to improve the model accuracy.

# In[ ]:


# Define the hyperparameters grid
param_grid = {
    'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 200, 400, 800, 1600],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion' : ['gini', 'entropy']
}

# Initialize RandomForestClassifier
random_forest_model = RandomForestClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train the grid search
grid_search.fit(X_train_combined, y_train_combined)

# Print the best hyperparameters found
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model
best_rf_model = grid_search.best_estimator_

# Evaluate the best model on testing set
print("Evaluation of the best RF model on the testing set:")
evaluate_model(best_rf_model, X_test_combined, y_test_combined)


# In[ ]:


# save the model
model_path = "/content/drive/MyDrive/multimodal project/models/Rf_GridSerach.pkl"
save_model(best_rf_model, model_path)


# In[149]:


# Load the saved model from the file
model_path = "/content/drive/MyDrive/multimodal project/models/Rf_GridSerach.pkl"
loaded_model = load_model(model_path)


# In[150]:


# Evaluate the best model on training set
print("Evaluation of the best RF model on the training set:")
evaluate_model(loaded_model, X_train_combined, y_train_combined)


# In[144]:


# Evaluate the best model on testing set
print("Evaluation of the best RF model on the testing set:")
evaluate_model(loaded_model, X_test_combined, y_test_combined)


# # SVM
# we will apply SVM Algorithm to train the model, and we will us Grid Search Algorithm with Cross Validation to improve the performance.

# In[ ]:


from sklearn.svm import SVC

# Initialize SVM classifier
svm_classifier = SVC()

# Training SVM model
train_model(svm_classifier, X_train_combined, y_train_combined)


# In[ ]:


# save the model
model_path = "/content/drive/MyDrive/multimodal project/models/SVM.pkl"
save_model(best_rf_model, model_path)


# In[151]:


# Load the saved model from the file
model_path = "/content/drive/MyDrive/multimodal project/models/SVM.pkl"
loaded_model = load_model(model_path)


# In[153]:


# evaluate SVM model on training set
print("Evaluation RF model on Training dataset:")
evaluate_model(loaded_model, X_train_combined, y_train_combined)


# In[154]:


# evaluate SVM model on Testing set
print("Evaluation RF model on testing dataset:")
evaluate_model(loaded_model, X_test_combined, y_test_combined)


# ## Grid Search for SVM

# In[ ]:


# Define the hyperparameters grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100],  # Penalty parameter C
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 2]  # Kernel coefficient
}

# Initialize SVM classifier
svm_classifier = SVC()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train the grid search
grid_search.fit(X_train_combined, y_train_combined)

# Print the best hyperparameters found
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model
best_svm_model = grid_search.best_estimator_

# Evaluate the best model on testing set
print("Evaluation of the best SVM model on the testing set:")
evaluate_model(best_svm_model, X_test_combined, y_test_combined)


# In[ ]:


# save the model
model_path = "/content/drive/MyDrive/multimodal project/models/SVM_GridSearch.pkl"
save_model(best_rf_model, model_path)


# In[158]:


# Load the saved model from the file
model_path = "/content/drive/MyDrive/multimodal project/models/SVM_GridSearch.pkl"
loaded_model = load_model(model_path)


# In[159]:


# Evaluate the best model on training set
print("Evaluation of the best SVM model on the training set:")
evaluate_model(loaded_model, X_train_combined, y_train_combined)


# In[160]:


# Evaluate the best model on testing set
print("Evaluation of the best SVM model on the testing set:")
evaluate_model(loaded_model, X_test_combined, y_test_combined)


# ## Using the model

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from IPython.display import Audio, display
import joblib
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import joblib
from skimage import feature


# In[ ]:


# Function to load a trained model from a file
def load_model(model_path):
    return joblib.load(model_path)

def play_audio(audio_file_path):
    display(Audio(audio_file_path, autoplay=True))

def Extract_voice_features(audio_file_path, scaler=None):
    # Load the audio file
    samples, sample_rate = librosa.load(audio_file_path, sr=None)

    # Extract features
    fo, fhi, flo, jitter, shimmer, nhr, hnr, rpde, dfa = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=9)

    # Calculate mean of each feature
    voice_features_mean = [fo.mean(), fhi.mean(), flo.mean(), jitter.mean(), shimmer.mean(), nhr.mean(), hnr.mean(), rpde.mean(), dfa.mean()]

    # Convert selected features to array
    voice = np.array(voice_features_mean)

    # If scaler is provided, scale the features
    if scaler:
        voice_scaled = scaler.fit_transform(voice.reshape(-1, 1)).reshape(1, -1)
    else:
        voice_scaled = voice.reshape(1, -1)

    return voice_scaled

# Function to extract HOG features from an image
def Extract_img_features(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (250, 250), fx=0.5, fy=0.5)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = feature.hog(img, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features.reshape(1, -1), img

def predict_parkinsons(audio_path, image_path, model):

  voice_features = Extract_voice_features(audio_path, scaler)

  img_features, img = Extract_img_features(image_path)

  # Concatenate voice and image features
  model_input = np.concatenate((voice_features, img_features), axis=1)

  # Make predictions
  prediction = model.predict(model_input)

  # Play the audio file
  play_audio(audio_path)

  # Display the image
  plt.imshow(img, cmap='gray')
  plt.axis('off')

  if prediction == 1:
    return "You are sick"
  else:
    return "You do not have Parkinson's disease"

# Initialize the StandardScaler
scaler = StandardScaler()


# In[ ]:


# Load the saved model from the file
model_path = "/content/drive/MyDrive/multimodal project/models/Rf.pkl"
loaded_model = load_model(model_path)


# In[ ]:


audio_path = "/content/drive/MyDrive/multimodal project/datasets/voice/records for testing/pd3.wav"
image_path = "/content/drive/MyDrive/multimodal project/datasets/drawings/images for Using/hc1.png"


# In[ ]:


predictions = predict_parkinsons(audio_path, image_path, loaded_model)
print(predictions)

