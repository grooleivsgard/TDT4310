
import gensim
from gensim import models, similarities
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pprint import pprint


def feature_terms(processed_poem):
    try:
        # create the vectorizer object (max_features is the number of words to keep)
        vectorizer = TfidfVectorizer(max_features=5)  # source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        
        # convert all poems to term-document matrix
        matrix = vectorizer.fit_transfor