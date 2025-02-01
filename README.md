# Classifying Love Speech in Norwegian Poetry

This project investigates binary text classification on a dataset of historical Norwegian poetry to detect expressions of love and compassion ("love speech"). Unlike most previous work that focused on negative sentiments such as hate speech, this study explores how positive sentiments are expressed and how they have evolved over more than 200 years of literary history.

## Table of Contents

- [Project Overview](#project-overview)
- [Approach and Methodology](#approach-and-methodology)
  - [Data Acquisition and Preprocessing](#data-acquisition-and-preprocessing)
  - [Model Architecture and Workflow](#model-architecture-and-workflow)
  - [Experimental Setup and Results](#experimental-setup-and-results)
- [Evaluation and Discussion](#evaluation-and-discussion)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [References](#references)

## Project Overview

- **Objective:**  
  Develop and evaluate machine learning models to classify whether a poem contains love speech.

- **Dataset:**  
  Approximately 684 Norwegian poems dating from the 1800s to today. Poems were scraped from [dikt.org](http://dikt.org) and stored in JSON format. The data includes tags (e.g., "Kjærlighet" for love) and additional metadata such as the decade of composition.

- **Models Evaluated:**  
  - **Logistic Regression:** Showed the most balanced performance with an overall accuracy of 85.19%.
  - **Multinomial Naïve Bayes:** Achieved an overall accuracy of 71.85%, with challenges in classifying the minority (love poems) class.
  - **NB-BERT (Transformer-based):** Reached an accuracy of 74.81% but struggled with the minority class.
  - **Random Baseline:** Used for comparison, with an accuracy of 46.67%.

## Approach and Methodology

### Data Acquisition and Preprocessing

1. **Scraping and Data Collection:**  
   - Poems were scraped from the website with a custom-built scraper.
   - Each poem was stored as a JSON file with properties such as `poem`, `tags`, and `title`.
   - Duplicate entries across categories were identified and consolidated.  

2. **Data Cleaning:**  
   - HTML breaks (`<br>`) were replaced with newline characters (`\n`).
   - Special Norwegian characters (æ, ø, å) were correctly decoded using `utf-8` encoding.
   - Preprocessing included tokenization, stemming, lemmatization, and stopword removal (with an extended stopword list to cover older Norwegian words).

3. **Data Transformation:**  
   - JSON files were converted to a CSV file with a new column, `is_love_poem`, indicating if a poem is tagged with "Kjærlighet".
   - The data was partitioned:
     - **For Logistic Regression and Naïve Bayes:** 80% training and 20% testing (using TF-IDF vectorization).
     - **For NB-BERT:** Split into 60% training, 20% validation, and 20% testing (using raw text).

4. **Handling Class Imbalance:**  
   - SMOTE was used to oversample the minority class (love poems) to balance the dataset.

### Model Architecture and Workflow

The project employs a modular architecture:
- **Data Loading and Cleaning:** A set of functions handles CSV conversion, stopword loading, and text cleaning.
- **Data Preparation:** The data is split, vectorized (for classical models), or directly fed to the NB-BERT model.
- **Model Training and Prediction:**  
  - Logistic Regression and Naïve Bayes are implemented using `sklearn`.
  - NB-BERT is implemented using HuggingFace's transformers API.
- **Result Visualization:** Classification reports and confusion matrices are generated to compare model performance.

**Workflow Diagram:**  
![Application Workflow Diagram](Images/flowchart.png)

### Experimental Setup and Results

- **Evaluation Metrics:**  
  Models were assessed using accuracy, precision, recall, and confusion matrices.

- **Results Summary:**  
  - **Logistic Regression:**  
    - Overall accuracy: 85.19%  
    - Non-love poems: Precision 88.6%, Recall 93.52%  
    - Love poems: Precision 66.67%, Recall 51.85%
  - **Multinomial Naïve Bayes:**  
    - Overall accuracy: 71.85%  
    - Lower performance on the love poem class (high false positives).
  - **NB-BERT:**  
    - Overall accuracy: 74.81%  
    - Struggled with minority class: only 14.81% recall on love poems.
  - **Random Baseline:**  
    - Accuracy: 46.67%

    
## Evaluation and Discussion

- **Observations:**  
  - Logistic Regression offered the most balanced performance.
  - All models faced challenges with the minority class due to class imbalance and the evolving language over two centuries.
  - Historical variations in language use affected the models' ability to capture love speech accurately.

- **Future Improvements:**  
  - Enhance preprocessing techniques to better capture historical language nuances.
  - Experiment with hybrid models or fine-tune training parameters (e.g., learning rate, number of epochs).
  - Increase dataset size for improved model reliability, especially for the minority class.

## Conclusion and Future Work

This study demonstrates that while conventional and advanced NLP models can distinguish love speech from other sentiments in historical Norwegian poetry, there are challenges—especially in handling class imbalance and linguistic evolution. Future work will focus on refining preprocessing steps and exploring more sophisticated modeling approaches to improve the detection of love speech in historical texts.

## References

- Kochmar et al., NLP and Text Classification Chapters
- Løvås, Logistic Regression for Text Classification
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- HuggingFace Transformers Documentation
- scikit-learn Documentation
- Imbalanced Learn Documentation

