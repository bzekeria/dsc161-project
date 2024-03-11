# Libraries
import pandas as pd
import json
import ast
import re
import nltk
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score

# Download NLTK resources
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')



# ---------------------------------------------------------------------------------- #

# 00_data
data = []

with open("data/case_with_all_sources_with_companion_cases_tag.jsonl", "r") as file:
    for line in file:
        data.append(json.loads(line))

df = pd.DataFrame(data)

df = df[(df["year"] >= 2010) & (df["year"] <= 2015)].reset_index(drop=True)

df.to_csv("data/super_scotus_raw.csv")



# ---------------------------------------------------------------------------------- #

## 01_data_cleaning
df = pd.read_csv("data/super_scotus_raw.csv")

## Convert dictionary-like strings to dictionaries

def parse_dictionary_value(value):
    # Replace NaN values with None
    value = re.sub(r'nan', 'None', str(value))
    # Remove unnecessary characters using regex
    value = re.sub(r'[^\x00-\x7F]+', '', value)
    # Remove words starting with "\"
    value = re.sub(r'\b\\[a-zA-Z0-9_]+\b', '', value)
    try:
        # Try to parse the string as a dictionary
        parsed_dict = ast.literal_eval(value)
        return parsed_dict
    except (SyntaxError, ValueError):
        return None

# List of columns containing dictionary-like strings
str_to_dict_cols = ["votes_detail", "scdb_elements", "convos", "oyez_summary", "justia_summary"]

for col in str_to_dict_cols:
    df[col] = df[col].fillna('{}')  # Fill null values with an empty dictionary
    df[col] = df[col].apply(parse_dictionary_value)

# Define delimiters for different sections
opinion_marker = "Opinion"
concurrence_marker = "Concurrence"
dissent_marker = "Dissent"

# Function to separate sections for a given text string
def separate_sections(text_string):
    # Find the start and end positions of each section
    opinion_start = text_string.find(opinion_marker)
    concurrence_start = text_string.find(concurrence_marker)
    dissent_start = text_string.find(dissent_marker)

    # Extract sections
    opinion_section = text_string[opinion_start + len(opinion_marker):concurrence_start].strip()
    concurrence_section = text_string[concurrence_start + len(concurrence_marker):dissent_start].strip()
    dissent_section = text_string[dissent_start + len(dissent_marker):].strip()

    # Store sections in a dictionary
    sections_dict = {"Opinion": opinion_section}

    # Check if concurrence and dissent sections exist and add them to the dictionary
    if concurrence_start != -1:
        sections_dict["Concurrence"] = concurrence_section
    if dissent_start != -1:
        sections_dict["Dissent"] = dissent_section

    return sections_dict

df["justia_sections"] = df["justia_sections"].apply(separate_sections)

# Convert float to int
df["win_side"] = df["win_side"].astype(int)

# Check if all of the columns that should be a dict, are indeed a dict
## Omitted the "convos" column as it didn't pass and its not needed for this project
test_cases = {
    "votes_detail": {"all_dicts": True},
    "scdb_elements": {"all_dicts": True},
    "oyez_summary": {"all_dicts": True},
    "justia_summary": {"all_dicts": True},
    "justia_sections": {"all_dicts": True}
}



# ---------------------------------------------------------------------------------- #

## 02_text_preprocessing
df = pd.read_csv("data/super_scotus_cleaned.csv")

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatization using NLTK
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Remove numbers
    non_numeric_tokens = [word for word in lemmatized_tokens if not word.isnumeric()]
    
    # Convert to lowercase
    lowercase_tokens = [word.lower() for word in non_numeric_tokens]
    
    preprocessed_text = ' '.join(lemmatized_tokens)
    
    return preprocessed_text

df['justia_summary'] = df['justia_summary'].apply(preprocess_text)

df['justia_sections'] = df['justia_sections'].apply(preprocess_text)

assert not any(char.isdigit() for char in df['justia_summary']), "Numbers are present in the preprocessed text of the 'justia_summary' column."

assert not any(char.isdigit() for char in df['justia_sections']), "Numbers are present in the preprocessed text of the 'justia_summary' column."

df[['justia_summary', 'justia_sections']].head()

df.to_csv("data/super_scotus_text_preprocessed.csv", index = False)

for col, expected_result in test_cases.items():
    non_dict_rows = []
    for idx, value in df[col].items():
        if not isinstance(value, dict):
            non_dict_rows.append(idx)
    assert len(non_dict_rows) == 0, f"Test case failed: {col} contains non-dictionary values"
    print(f"Test case passed: {col} contains only dictionary values")

assert set().union(*(d.keys() for d in df["justia_sections"])) == {'Concurrence', 'Dissent', 'Opinion'}

## Subset data

df = df[['votes_detail', 'win_side','title', 'petitioner', 'respondent', 'year', 'docket_no', 'justia_summary', 'justia_sections']]


## Export data

df.to_csv("data/super_scotus_cleaned.csv", index = False)



# ---------------------------------------------------------------------------------- #

## 03_model

df = pd.read_csv("data/super_scotus_text_preprocessed.csv")

X_text = df['justia_sections']  # Textual content
X_metadata = df.drop(columns=['justia_sections', 'win_side'])  # Metadata
y = df['win_side']  # Target variable

X_text_train, X_text_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
    X_text, X_metadata, y, test_size=0.2, random_state=42)

# Text preprocessing: TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer() 
X_text_train_tfidf = tfidf_vectorizer.fit_transform(X_text_train)
X_text_test_tfidf = tfidf_vectorizer.transform(X_text_test)

# Combine TF-IDF features with metadata features
X_train_text_tfidf_df = pd.DataFrame(X_text_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
X_test_text_tfidf_df = pd.DataFrame(X_text_test_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# X_train = pd.concat([X_meta_train.reset_index(drop=True), X_train_text_tfidf_df], axis=1)
# X_test = pd.concat([X_meta_test.reset_index(drop=True), X_test_text_tfidf_df], axis=1)

X_train = X_train_text_tfidf_df

X_test = X_test_text_tfidf_df

naive_bayes = MultinomialNB()
logistic_regression = LogisticRegression()

# Cross-validation to evaluate models
naive_bayes_cv_scores = cross_val_score(naive_bayes, X_train, y_train, cv=5)
logistic_regression_cv_scores = cross_val_score(logistic_regression, X_train, y_train, cv=5)

print('Naive Bayes Cross-Validation Scores:', naive_bayes_cv_scores)
print('Naive Bayes Mean Cross-Validation Score:', naive_bayes_cv_scores.mean())
print('Logistic Regression Cross-Validation Scores:', logistic_regression_cv_scores)
print('Logistic Regression Mean Cross-Validation Score:', logistic_regression_cv_scores.mean())

# Select the best model based on cross-validation scores
best_model = 'Naive Bayes' if naive_bayes_cv_scores.mean() > logistic_regression_cv_scores.mean() else 'Logistic Regression'

# Fit the selected model on the entire training dataset
if best_model == 'Naive Bayes':
    best_model = naive_bayes.fit(X_train, y_train_encoded)
else:
    best_model = logistic_regression.fit(X_train, y_train)

test_score = best_model.score(X_test, y_test)
print(f'Test Score of Best Model ({best_model}): {test_score}')

y_pred = best_model.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(precision)

print(recall)



