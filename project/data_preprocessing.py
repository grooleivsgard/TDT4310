import os
import pickle
import pandas as pd
import json
import spacy
import re
import models as md
from pathlib import Path
from transformers import BertTokenizer, AutoTokenizer, AutoModelForTokenClassification, pipeline, TFAutoModelForSequenceClassification, optimization_tf, AutoConfig
from models import model_name

tokenizer_nb = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base-ner")
nlp_nb = pipeline("ner", model=model_name, tokenizer=tokenizer_nb)
nlp = spacy.load('nb_core_news_md') #skriv denne i terminal "python -m spacy download nb_core_news_md"


# ----------------- Data Loading -----------------
def load_csv(filename):
    if os.path.exists(filename):
        df = add_love_poem_column(filename)
        return df
    else:
        print("No CSV file found.")
        return None

def json_to_csv(directory, filename):
    
    data = []
    
    print("Creating CSV from JSON files...")
    for file in os.listdir(directory):
        if file.endswith('.json'):
            file_path = os.path.join(directory, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                # load file content
                json_data = json.load(f)
                # append file content to data list
                data.append({
                    'title': json_data['title'],
                    'poem': json_data['poem'],
                    'tags': ', '.join(json_data['tags'])  # join tags with comma if there are multiple
                })
    
    # convert list to dataframe
    df = pd.DataFrame(data)

    print("Converting DataFrame to CSV...")
    # convert dataframe to csv
    df.to_csv(filename, index=False)
    print("CSV successfully created!")
    
# add a new column in the CSV as "is_love_poem". for each poem, check if the item contains the tag "Kjærlighet" and assign 1 to is_love_poem if it does, 0 otherwise
def add_love_poem_column(filename):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)
    # Check if 'is_love_poem' column already exists
    if 'is_love_poem' not in df.columns:
        df['is_love_poem'] = df['tags'].apply(lambda x: 1 if 'Kjærlighet' in x else 0)
        # Save the updated DataFrame back to the CSV file
        df.to_csv(filename, index=False)
        print("Added 'is_love_poem' column to the CSV file.")

    return df

# not tested
def drop_column(df, column_name):
    if column_name in df.columns:
        df.drop(column_name, inplace=True)
        # df.drop(column_name, axis=1, inplace=True)
        print(f"{column_name} column is deleted.")
    else:
        print(f"{column_name} column does not exist in the DataFrame.")
    
def create_csv(df, filename):

    print("Creating CSV file...")

    if 'processed_poem' in df.columns:
        df.drop('processed_poem', axis=1, inplace=True)
        print("Processed poem column is deleted.")
    print("Saving DataFrame to CSV.")
    df.to_csv(filename, index=False)
    print(f"DataFrame saved to {filename}") 

def check_empty_fields(filename):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Check for any empty fields in the DataFrame
    if df.isna().any().any():
        print("There are empty fields in the following columns:")
        # List columns that contain at least one empty field
        empty_columns = df.columns[df.isna().any()].tolist()
        print(empty_columns)
    else:
        print("There are no empty fields in the CSV file.")
    
def add_decade_column(filename):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)
    # Check if 'decade' column already exists
    if 'decade' not in df.columns:
        # finds the pattern '1800-tallet', '1900-tallet', or '2000-tallet' in the tags column, and assign the value 1800, 1900 or 2000 to the decade column
        df['decade'] = df['tags'].apply(lambda x: 1800 if '1800-tallet' in x else (1900 if '1900-tallet' in x else (2000 if '2000-tallet' in x else 0)))
        # Save the updated DataFrame back to the CSV file
        df.to_csv(filename, index=False)
        print("Added 'decade' column to the CSV file.")
    return df

# ----------------- Preprocessing -----------------

# create or load a list of stop words
def load_stopwords():
    filename = 'stopwords.txt'

    spacy_stopwords = spacy.lang.nb.stop_words.STOP_WORDS # load the stop words from spaCy
    additional_stopwords = ['paa', 'saa', 'naa', 'naar', 'vaar', 'hvad', 'blev', 'op', 'vor', 'ud', 'af', 'nu', 'mig', 'sig', 'sit', 'mod', 'ind', 'aa', 'å', 'én', 'eg']
    
    # create the stopwords list if it doesn't exist
    if not os.path.exists(filename):
         with open(filename, 'w', encoding='utf-8') as file:
            for word in spacy_stopwords:
                file.write(word + '\n')
            for word in additional_stopwords: # add additional stopwords
                file.write(word + '\n')  
            print("Stopword list created: stopwords.txt")
    else:
        # read the stopwords from the file if it exists
        with open(filename, 'r', encoding='utf-8') as file:
            all_stopwords = [line.strip() for line in file if line.strip()]  # remove any empty lines
    
    return all_stopwords

# clean the text data
def clean_text(text: str, stopwords: set) -> str:

    # tokenizes, POS tags, dependency parsing
    text = re.sub(r'\s+', ' ', text.lower().replace('\n', ' '))
    doc = nlp(text)
    
    # lemmatize the tokens, remove stopwords, punctuation, and short words (<= 1 character)
    cleaned_text = ' '.join([token.lemma_ for token in doc 
                             if token.lemma_ not in stopwords 
                             and not token.is_punct and len(token.lemma_) > 1])
    
    # replace newlines with a space
    cleaned_text = cleaned_text.replace('\n', ' ')

    # replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text

# retrieve entities
def retrieve_entities(tags):
    for tag in tags['tags']:
        print(f"Tag: {tag}")
        print(f"Entities: {md.nlp_nb(tag)}")


        