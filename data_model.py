#import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# import dataframe
df = pd.read_csv("data/cleaned_data.csv")

# Tokenize and vectorize skills using spaCy
df['skillset'] = df['skillset'].apply(lambda x: nlp(x).vector)

# Encode job categories
df['job_role'] = df['job_role'].astype('category').cat.codes

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['skillset'].tolist(), df['job_role'], test_size=0.2, random_state=42)

# Choose a model (Decision Tree Classifier)
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train.tolist(), y_train)

# Make predictions
predictions = model.predict(X_test.tolist())

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

"""
# Assuming 'new_skills' is a list of skill vectors for a new set of skills
new_skills = [nlp(skill).vector for skill in ["skill1", "skill2", "skill3"]]

# Make predictions for the new set of skills
predicted_category = model.predict([new_skills])[0]

# Decode the predicted category back to the original label
predicted_category_label = df['job_role'].cat.categories[predicted_category]

print(f"Predicted Job Category: {predicted_category_label}")

"""