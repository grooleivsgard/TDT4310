from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk


def calc_freq_dist(poems):
    freq_dist = Counter()

    for text in poems:
        words = text.split()
        freq_dist.update(words)

    return freq_dist

def filter_poems_by_tags(df, tag_1, tag_2):
    # Filter for poems that include the specific tag of interest and are from a specified century
    result = df['tags'].apply(lambda tags: tag_1 in tags and tag_2 in tags)
    # return a list of the titles of the poems that have both tags
    if result.any():
        poem_titles = df[result]['title'].tolist()
        return poem_titles
    else:
        return "No poems found with the specified tags"

def compare_tf_idf(df):
    # Tags of interest
    interest_tag = 'Vennskap'
    century_tags = ['Dikt skrevet på 1800-tallet', 'Dikt skrevet på 1900-tallet', 'Dikt skrevet på 2000-tallet']

    results = {}

    # Process each set of tags
    for century_tag in century_tags:
        filtered_poems = filter_poems_by_tags(df, interest_tag, century_tag)
        if not filtered_poems.empty:
            cleaned_poems = filtered_poems['cleaned_poem'].tolist()
            common_words = calc_freq_dist(cleaned_poems)
            results[century_tag] = common_words
        else:
            print(f"No poems found with tags {interest_tag} and {century_tag}")

    # Output the results
    for century_tag, words in results.items():
        print(f"Common words in '{interest_tag}' poems from {century_tag}:\n", words.tolist())

def testing_stuff(df):
    # Calculate frequency distribution of words in all poems
    #freq_dist = calc_freq_dist(df['processed_poem'])
    #print(f"Freq dist: {freq_dist.most_common(10)}")

    #tf_idf = calc_tf_idf(df['processed_poem'])
    # print(f"tf idf: {tf_idf.tolist()}")

    # Compare common words from poems with different tags
    # compare_tf_idf(df)

    # Calculate frequency distribution of words in a poem
    # print(processed_poems[0])

    '''for poem in filter_poems_by_tags(df, 'Kjærlighet', 'Dikt skrevet på 1800-tallet')['processed_poem']:
        print(Counter(poem).most_common(5))'''

    
    ''' words = [token.text for token in processed_poems if not token in stopwords]
    print(Counter(words).most_common(5))'''
    
    # print the title of the poem that has both 'Kjærlighet' and 'Vennskap' tags
    #print(filter_poems_by_tags(df, 'Kjærlighet', 'Sorg'))

    tags = {
    "tags": [
        "Vår",
        "Vestlandsviser",
        "Vilhelm Krag",
        "Dikt skrevet på 1800-tallet",
        "Alle dikt alfabetisk"
        ]
    }

   #  dp.retrieve_entities(tags)

    # Add a new column to the dataframe that indicates whether a poem is a love poem
    # dp.add_love_poem_column(df)

    # print(df.head())

    # Get the number of poems in the DataFrame
    print(f"Number of poems: {len(df)}")
    # Get the number of poems with tag 'Kjærlighet'
    print(f"Number of poems with tag 'Kjærlighet': {len(df[df['tags'].apply(lambda x: 'Kjærlighet' in x)])}")