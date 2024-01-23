"""
extract skill-set from about, experience, and certificate columns

"""

import pandas as pd
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_skills(text):
    doc = nlp(text)
    # Extract unique lemmatized tokens as skills
    #skills = list(set(token.lemma_ for token in doc if token.pos_ == "NOUN" or token.pos_ == "PROPN"))
    ner_skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    print(ner_skills)
    return ner_skills

# import dataframe
df = pd.read_csv("data/concatenated_file.csv")

# Create a new 'Skillset' column
df['skillset'] = df['about'].fillna('') + ' ' + df['experience'].fillna('') + ' ' + df['certifications'].fillna('')

# Apply the extract_skills function to each row in the 'Skillset' column
df['skillset'] = df['skillset'].apply(extract_skills)

# Optionally, you can convert the 'Skillset' column to strings if you want a comma-separated list
df['skillset'] = df['skillset'].apply(lambda skills: ', '.join(skills))

# Display the DataFrame
print(df)

# write back to a new file
output_file = 'data/cleaned_data.csv'

# Write DataFrame to CSV
df[['job_role', 'skillset']].to_csv(output_file, index=False)
