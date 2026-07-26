#!/usr/bin/env python
# coding: utf-8

# # Parkinson Disease Early Detection
# in this project, we will use machine learning algorithms to detect parkinson disease.
# <b> in the begining, </b> we will install and import necessary modules and read our dataset.

# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[51]:


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
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import librosa


# In[52]:


# Specify the file path
file_path = "/content/drive/MyDrive/multimodal project/datasets/voice/Parkinsson disease.csv"

# Read data from the CSV file
data = pd.read_csv(file_path)


# In[53]:


# show the first five samples
data.head()


# # Data Distribution
# we will visualize data distribution, this proess help us to understand dataset deeply.
# 

# In[ ]:


plt.figure(figsize=(15, 8))

# Plot the distribution of 'MDVP:Fo(Hz)'
sns.histplot(data['MDVP:Fo(Hz)'], bins=20, kde=True, color='blue')

# Set the title and labels
plt.title('Distribution of MDVP:Fo(Hz)')
plt.xlabel('MDVP:Fo(Hz)')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

# In[54]:


# show information about dataset
data.info()


# In[55]:


# show statistical description for dataset
data.describe()


# In[56]:


# detect the number of missing values in each colum
missing_values = data.isnull().sum()
missing_values


# In[57]:


data['status'].value_counts()


# In[58]:


# Drop unnecessary column
data = data.drop('name', axis=1)

# Calculate z-score for each feature
z_scores = (data - data.mean()) / data.std()

# Remove outliers using the standard threshold = 3
data = data[(z_scores < 3).all(axis=1)]

# Check that the number of samples equals 181 and the number of features equals 13
print("Number of samples after removing outliers:", data.shape[0])
print("Number of features after removing outliers:", data.shape[1])


# In[59]:


# Calculate the correlation coefficient between each pair of features
correlation_matrix = data.corr()

# Identify columns with strong correlations (more than 0.8 or less than -0.8)
strong_correlation = np.where(np.abs(correlation_matrix) > 0.8)

# Drop the column if it has a strong correlation with any other column
for col1, col2 in zip(*strong_correlation):
    if col1 != col2 and col1 < col2:
        if col1 in data.columns and col2 in data.columns:
            data = data.drop(data.columns[col2], axis=1)


# In[60]:


data.shape


# # Balancing the data
# We noticed that the data we have is unbalanced, and in order for the result that we will obtain to be fair and so that the algorithm is not biased, we will balance the data using one of the data augmentation techniques so that the number of patient samples becomes equal to the number of healthy people’s samples.

# In[114]:


# Shuffle the dataset
data = shuffle(data, random_state=42)


# In[115]:


# Separate features (X) and target variable (y)
X = data.drop('status', axis=1)
y = data['status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("x_train shape is: {}".format(X_train.shape))
print("y_train shape is: {}".format(y_train.shape))
print("X_test shape is: {}".format(X_test.shape))
print("y_test shape is: {}".format(y_test.shape))


# In[116]:


# Display class distribution before balancing for training set
print("Class distribution before balancing for training set:")
print(y_train.value_counts())


# In[117]:


# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Display class distribution after balancing for training set
print("\nClass distribution after balancing for training set:")
y_train_balanced.value_counts()


# In[118]:


# Calculate class distribution after balancing
class_distribution = pd.Series(y_train_balanced).value_counts()

# Plot the class distribution
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.title('Distribution of Classes after Balancing')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[119]:


X_train_balanced.head()


# # feature selection
# Here we will use one of the feature selection techniques to choose a number of features that will be input to the machine learning algorithms, as we will not train the algorithm on all the features in the dataset, but rather we will choose the best of these features.

# In[120]:


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

# In[121]:


# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the features
X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)


# # Reduce data dimensionality
# we will reduce the number of features using PCA algorithm.

# In[122]:


# Step 2: Compute the covariance matrix
cov_matrix = np.cov(X_train.T)

# Step 3: Compute the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Select the top eigenvectors based on the explained variance
total_variance = sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance
explained_variance_ratio_cumsum = np.cumsum(explained_variance_ratio)

# Choose the number of components to retain based on the explained variance ratio
n_components = 9  # You can adjust this based on the desired explained variance

# Step 5: Transform the original dataset into the new feature space
pca = PCA(n_components=n_components)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[123]:


# Get the names of the selected features
selected_feature_indices = pca.components_

# Assuming you have a list of feature names, you can use the selected indices to get the names
selected_features = [selected_features[i] for i in range(n_components)]


# In[124]:


selected_features


# in each algorithm we will train and evaluate the model and visualize accuracy and error for test data in each it and draw confusion matrix for testing data.
# # Random Forest Algorithm
# 

# In[125]:


# Initialize the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)

# Train the Random Forest model
random_forest_model.fit(X_train, y_train_balanced)


# In[126]:


# Predict the labels for the training set
y_pred_test = random_forest_model.predict(X_train)

# Calculate accuracy, precision, recall, and f1_score
accuracy_test_rf = accuracy_score(y_train_balanced, y_pred_test)
precision_test = precision_score(y_train_balanced, y_pred_test)
recall_test = recall_score(y_train_balanced, y_pred_test)
f1_test = f1_score(y_train_balanced, y_pred_test)

# Display the calculated metrics
print(f'Accuracy (train): {accuracy_test_rf}')
print(f'Precision (train): {precision_test}')
print(f'Recall (train): {recall_test}')
print(f'F1 Score (train): {f1_test}')

# Plot confusion matrix
conf_matrix_test = confusion_matrix(y_train_balanced, y_pred_test)
sns.heatmap(conf_matrix_test, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix (train Set)')
plt.show()


# In[127]:


# Predict the labels for the test set
y_pred_test = random_forest_model.predict(X_test)

# Calculate accuracy, precision, recall, and f1_score
accuracy_test_rf = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

# Display the calculated metrics
print(f'Accuracy (Test): {accuracy_test_rf}')
print(f'Precision (Test): {precision_test}')
print(f'Recall (Test): {recall_test}')
print(f'F1 Score (Test): {f1_test}')

# Plot confusion matrix
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
sns.heatmap(conf_matrix_test, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.show()


# # with cross validation

# In[128]:


from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib


# In[129]:


# Combine training and testing data
X_combined = np.concatenate((X_train, X_test), axis=0)
y_combined = np.concatenate((y_train_balanced, y_test), axis=0)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the features
X_combined_scaled = scaler.fit_transform(X_combined)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Define Leave-One-Subject-Out Cross-Validation
loo = LeaveOneOut()

# Initialize a list to store the accuracy scores
train_accuracy_scores = []
test_accuracy_scores = []

# Perform Leave-One-Subject-Out Cross-Validation
for train_index, test_index in loo.split(X_combined_scaled):
    X_train_loo, X_test_loo = X_combined_scaled[train_index], X_combined_scaled[test_index]
    y_train_loo, y_test_loo = y_combined[train_index], y_combined[test_index]

    # Fit the Random Forest Classifier to the training data
    rf_classifier.fit(X_train_loo, y_train_loo)

    # Predict the labels for the training data
    y_train_pred_loo = rf_classifier.predict(X_train_loo)

    # Calculate the training accuracy score for this iteration
    train_accuracy = accuracy_score(y_train_loo, y_train_pred_loo)
    train_accuracy_scores.append(train_accuracy)

    # Predict the labels for the test data
    y_test_pred_loo = rf_classifier.predict(X_test_loo)

    # Calculate the test accuracy score for this iteration
    test_accuracy = accuracy_score(y_test_loo, y_test_pred_loo)
    test_accuracy_scores.append(test_accuracy)

# Calculate the average training and testing accuracy across all iterations
average_train_accuracy = sum(train_accuracy_scores) / len(train_accuracy_scores)
average_test_accuracy = sum(test_accuracy_scores) / len(test_accuracy_scores)

# Print the average training and testing accuracy
print("Average Training Accuracy:", average_train_accuracy)
print("Average Testing Accuracy:", average_test_accuracy)


# In[130]:


# Train the model on the entire dataset
# rf_classifier.fit(X_combined_scaled, y_combined)
rf_classifier.fit(X_train, y_train_balanced)

X_train, y_train_balanced
# Save the trained model to a file
joblib.dump(rf_classifier, '/content/drive/MyDrive/multimodal project/models/voice_modell.pkl')


# ## Using the model

# In[131]:


import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from IPython.display import Audio, display
import joblib


# In[132]:


def play_audio(audio_file_path):
    display(Audio(audio_file_path, autoplay=True))


# In[133]:


def predict_parkinsons(audio_file_path, loaded_model, scaler=None):
    # Load the audio file
    samples, sample_rate = librosa.load(audio_file_path, sr=None)

    # Extract features
    fo, fhi, flo, jitter, shimmer, nhr, hnr, rpde, dfa = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=9)

    # Calculate mean of each feature
    features_mean = [fo.mean(), fhi.mean(), flo.mean(), jitter.mean(), shimmer.mean(), nhr.mean(), hnr.mean(), rpde.mean(), dfa.mean()]

    # Convert selected features to array
    voice = np.array(features_mean)

    # If scaler is provided, scale the features
    if scaler:
        voice_scaled = scaler.fit_transform(voice.reshape(-1, 1)).reshape(1, -1)
    else:
        voice_scaled = voice.reshape(1, -1)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(voice_scaled)

    # Play the audio file
    play_audio(audio_file_path)

    if prediction == 1:
        return "You are sick"
    else:
        return "You do not have Parkinson's disease"


# In[134]:


# Load the saved model from the file
loaded_model = joblib.load('/content/drive/MyDrive/multimodal project/models/voice_modell.pkl')

# Initialize the StandardScaler
scaler = StandardScaler()


# In[146]:


# Usage example:
audio_file_path = '/content/drive/MyDrive/multimodal project/datasets/voice/records for testing/hc6.wav'
predictions = predict_parkinsons(audio_file_path, loaded_model, scaler)
print(predictions)


# In[ ]:




