import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import pickle

# Load your training data (replace 'your_training_data.csv' with your actual file)
df = pd.read_csv('data/cleaned_data.csv')

# Define the pipeline with transformations
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('svd', TruncatedSVD(n_components=50)),
    ('scaler', MinMaxScaler()),
    ('svm', SVC(kernel='linear'))
])

le = LabelEncoder()
df['job_role_encoded'] = le.fit_transform(df['job_role'])

# Fit the pipeline on your training data
X_train = df['Skillset']  # Assuming 'Skillset' is the column with skills
y_train = df['job_role_encoded']   # Assuming 'job_role' is the target column

pipeline.fit(X_train, y_train)

# Save the trained model and vectorizer as pickle files
with open('svm_model_.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

# Optionally, save the TF-IDF vectorizer separately (useful for preprocessing in Flask)
with open('tfidf_vectorizer_.pkl', 'wb') as vectorizer_file:
    pickle.dump(pipeline.named_steps['tfidf'], vectorizer_file)

with open('label_encoder.pkl', 'wb') as labels:
    pickle.dump(le, labels)