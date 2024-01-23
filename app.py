from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse

app = Flask(__name__)

# Load the pre-trained SVM model and TF-IDF vectorizer
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Instantiate LabelEncoder
le = LabelEncoder()

# Load the fitted LabelEncoder from training
with open('label_encoder.pkl', 'rb') as label_encoder_file:
    le.classes_ = pickle.load(label_encoder_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the user input
        user_input = request.form['user_input']

        # Vectorize the user input using the TF-IDF vectorizer
        input_vector = tfidf_vectorizer.transform([user_input])

        # Check if the matrix is sparse and convert to dense if necessary
        if issparse(input_vector):
            input_vector = input_vector.toarray()

 
        # Make a prediction using the SVM model
        prediction = svm_model.predict(user_input)

        # Map the prediction to a job category or skillset (customize as needed)
        #job_category = map_prediction_to_category(prediction)
        predicted_label = le.inverse_transform(prediction)

        return render_template('result.html', job_category=predicted_label)



if __name__ == '__main__':
    app.run(debug=True)
