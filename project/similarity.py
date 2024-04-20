
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
        matrix = vectorizer.fit_transform(processed_poem) # fit_transform returns a sparse matrix
        
        # create dataframe of common words count
        terms = vectorizer.get_feature_names_out()
        df_vectorizer = pd.DataFrame(data = matrix.toarray(), columns = terms)

        # Find the most important words by score
        term_importance = df_vectorizer.sum().sort_values(ascending=False)
        df_vectorizer_sorted = df_vectorizer[term_importance.index]

        return df_vectorizer_sorted.columns
    
    except Exception as e:
        raise Exception(f"Error: {e}")


def lsi_model(processed_poem):
    # lsi model reduces the dimensionality of the tf-idf matrix so similar poems can be grouped together

    tokenized_poems = [poem.split() for poem in processed_poem]
    
    # the dictionary maps each unique word to an integer ID by converting input text to a list of words and then pass it to the corpora.Dictionary() object
    dictionary = gensim.corpora.Dictionary(tokenized_poems)

    # convert each poem to a bag of words representation 
    corpus = []
    for poem in tokenized_poems:
        corpus.append(dictionary.doc2bow(poem))
    
    # initialise the tf-idf model using the bows of the poems
    tfidf = models.TfidfModel(corpus)

    # latent dimensions - the number of topics to extract
    num_topics = 5

    # create the LSI model using a tf-idf representation of the corpus
    lsi = models.LsiModel(tfidf[corpus], id2word=dictionary, num_topics=num_topics)

    print(lsi.print_topics())

    # model similarity between poems
    index = similarities.MatrixSimilarity(lsi[corpus])


    
