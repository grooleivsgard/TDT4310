import pandas as pd
import nltk
import spacy
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load('nb_core_news_md')

def get_poems_from_files(directory: str):
    poems = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    poems.append(data['poem'])
                except json.JSONDecodeError:
                    print(f"Error reading from JSON file. It might be corrupt or incomplete: {file_path}")
    return poems  # Just return the poems list here.

def get_common_words(poems):
    with open('norwegian_stop_words.txt', 'r', encoding='utf-8') as file:
        norwegian_stop_words = [line.strip() for line in file.readlines()]
    try:
        vectorizer = TfidfVectorizer(analyzer='word', stop_words=norwegian_stop_words, max_features=100)
        word_matrix = vectorizer.fit_transform(poems)
        tokens = vectorizer.get_feature_names_out()
        df_vectorizer = pd.DataFrame(data=word_matrix.toarray(), columns=tokens)
        word_importance = df_vectorizer.sum().sort_values(ascending=False)
        return word_importance.index.tolist()  # Return the list of column names sorted by importance
    except Exception as e:
        raise RuntimeError(f"Error: {e}")

# Example usage:
poems = get_poems_from_files('poems')
print(get_common_words(poems))


def clean_text(text: str) -> str:
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if len(token.text) > 1])

def processing(comments: list) -> list:
    return [clean_text(comment) for comment in comments]
