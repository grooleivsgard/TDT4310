
import pandas as pd
import json
import spacy
import data_preprocessing as dp
import os
import models as md
import similarity as sim
from data_preprocessing import nlp_nb
import matplotlib.pyplot as plt


def load_data():
    directory = 'poems'
    filename = 'poems.csv'

    # check if the csv file exists
    if not os.path.exists(filename):
        # Create a DataFrame from the JSON files
        df = dp.json_to_csv(directory, filename)

    # Load data as DataFrame
    df = dp.load_csv(filename)

    # Create a list of stop words if it doesn't exist
    stopwords = dp.load_stopwords()

    # check if the processed_poem column exists
    if 'processed_poem' not in df.columns:
        df['processed_poem'] = df['poem'].apply(lambda x: dp.clean_text(x, stopwords))

    return df


def load_classifier(df):
    poems_train, poems_test, y_train, y_test = md.split_data(df) # split into training and testing sets
    X_train, X_test = md.get_vectorizer(poems_train, poems_test) # get the tf-idf matrices
    X_train_oversampled, y_train_oversampled = md.oversample(X_train, y_train) # oversample the training data
    md.plot_distribution(y_train_oversampled) # plot the distribution of classes
    return md.train_classifier(X_train_oversampled, X_test, y_train_oversampled , y_test) # train and predict

def load_bert(df):
    X_train, X_test, X_val, y_train, y_test, y_val = md.split_data(df, with_validation_set=True)
    md.train_bert(X_train, X_test, X_val, y_train, y_test, y_val)

def load_similarity(df):

    # get the top tf-idf 20 term from all poems and love poems
    top_terms = sim.feature_terms(df['processed_poem'])
    top_love_terms = sim.feature_terms(df[df['is_love_poem'] == 1]['processed_poem'])
    print(top_terms)
    print(top_love_terms)
    sim.lsi_model(df['processed_poem'])

    # top words overall: ['min', 'din', 'liv', 'hvor', 'all', 'hjerte', 'stå', 'hver', 'dig','mit']
    # top love words: ['min', 'din', 'elske', 'kjærlighet', 'hjerte', 'liv', 'alltid', 'mit', 'stå', 'terje']
    

    # get the terms that are in the love poems but not in the top terms
    love_query = []
    for term in top_love_terms:
        if term not in top_terms:
            love_query.append(term)
    
    # love vector: ['elske', 'kjærlighet', 'hjerte']
    print(love_query)
    return love_query
    
def main():

    df = load_data()
    dp.load_stopwords()
    load_classifier(df)

    # print poem and processed poem columns from poem 2 to 6
    #print(df[['poem', 'processed_poem']].iloc[2:6])


    # -- machine learning models --
    # undersampled_df = md.undersample(df)
    # load_bert(undersampled_df)
    #load_similarity(df)
    
if __name__ == "__main__":
    main()
