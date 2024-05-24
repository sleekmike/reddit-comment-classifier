import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MaxAbsScaler
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import cleaner 


# ================================================> XGBoost Classifier <====================================================== #
# Load the datasets (Preprocessed, Cleaned, and Labelled)
df = pd.read_csv('../data/training_0-2000.csv')

# Preprocess and Clean Data
df = cleaner(df)

# Encode labels
label_encoder = LabelEncoder()
df['encoded_labels'] = label_encoder.fit_transform(df['Label'])

# Check the mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(label_mapping)

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(df['comments'], df['encoded_labels'], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Vectorize the text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Scale the features using MaxAbsScaler
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train_tfidf)
X_val_scaled = scaler.transform(X_val_tfidf)
X_test_scaled = scaler.transform(X_test_tfidf)

# Handle imbalanced classes using SMOTE
smote = SMOTE(random_state=42, k_neighbors=2)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Initialize XGBoost classifier and set up grid search
clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
parameters = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Define StratifiedKFold for cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-Validation and Hyperparameter Tuning
grid_search = GridSearchCV(clf, parameters, cv=kf, scoring='accuracy')
grid_search.fit(X_train_resampled, y_train_resampled)

# Best estimator
best_clf = grid_search.best_estimator_

# Predict on the validation set
y_pred_val = best_clf.predict(X_val_scaled)

# Evaluate the classifier on the validation set
print("Best Parameters for XGBoost:", grid_search.best_params_)
print("XGBoost Validation Classification Report")
#print(classification_report(y_val, y_pred_val, target_names=label_encoder.classes_))
print(classification_report(y_val, y_pred_val,target_names=label_encoder.classes_, labels=[0, 1, 2], zero_division=1))
print("XGBoost Validation Confusion Matrix")
print(confusion_matrix(y_val, y_pred_val))
print("XGBoost Validation Accuracy Score:", accuracy_score(y_val, y_pred_val))

# Predict on the test set
y_pred_test = best_clf.predict(X_test_scaled)

# Evaluate the classifier on the test set
print("XGBoost Test Classification Report")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, labels=[0, 1, 2], zero_division=1))
print("XGBoost Test Confusion Matrix")
print(confusion_matrix(y_test, y_pred_test))
print("XGBoost Test Accuracy Score:", accuracy_score(y_test, y_pred_test))

# Save the trained XGBoost model
joblib.dump(best_clf, './models/xgboost_model.pkl')
# Save the TfidfVectorizer
joblib.dump(tfidf_vectorizer, './models/xgboost_tfidf_vectorizer.pkl')
# Save the LabelEncoder
joblib.dump(label_encoder, './models/xgboost_label_encoder.pkl')