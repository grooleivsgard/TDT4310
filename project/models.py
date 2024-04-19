import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import BertTokenizer, AutoTokenizer, AutoModelForTokenClassification, pipeline, TFAutoModelForSequenceClassification, optimization_tf, AutoConfig
import tensorflow as tf # provided by Hugging Face
import math
from sklearn.metrics import classification_report

model_name = "NbAiLab/nb-bert-base"

# vurder å fjerne:
tokenizer_nb = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base-ner")
nlp_nb = pipeline("ner", model=model_name, tokenizer=tokenizer_nb)


# undersample to balance the classes
def undersample(df):
    # separate majority (non-love poems) and minority (love poems) classes
    df_majority = df[df.is_love_poem==0]
    df_minority = df[df.is_love_poem==1]
    
    # downsample majority class (for example, to the same size as the minority class)
    df_majority_downsampled = df_majority.sample(300, random_state=42)
    
    # combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    
    return df_downsampled

# turn test and train data into tf-idf matrices
def get_vectorizer(train_data, test_data):

    vectorizer = TfidfVectorizer()

    # convert all poems to term-document matrix
    X_train = vectorizer.fit_transform(train_data) # fit_transform creates the tf-idf matrix based on the training data
    X_test = vectorizer.transform(test_data) # transform uses the same vectorizer to create the tf-idf matrix for the test data

    return X_train, X_test
    

# split data into training and testing sets
def split_data(df, with_validation_set=False):

    # X is the processed poem column (text), y is the is_love_poem column (0 or 1)
    # training data is 80% of the data, testing data is 20%
    X_train, X_test, y_train, y_test = train_test_split(df['poem'], df['is_love_poem'], test_size=0.2, random_state=42) # IMPORTANT: if using MultinomialNB or LogsiticRegression, the input data must be 'processed_poem' instead of 'poem'

    # split the test data into test and validation sets
    if with_validation_set:
        # split the training data into training (now 60% of the total data) and validation sets (20% of the total data)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1) # 0.25 x 0.8 = 0.2
        print("Data is split into training, testing and validation sets.")
        # print a sample from each set
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    return X_train, X_test, y_train, y_test


def train_classifier(X_train, X_test, y_train, y_test):
    
    # create the naive bayes classifier object
    clf = MultinomialNB() # Change to LogisticRegression(C=1) to use logistic regression with C=1 (default)

    # train the classifier
    clf.fit(X_train, y_train)

    # predict the test data
    predictions = clf.predict(X_test)
    
    '''
    # calculate the accuracy
    accuracy = accuracy_score(y_test, predictions)

    # calculate the precision
    precision = precision_score(y_test, predictions)

    # calculate the recall
    recall = recall_score(y_test, predictions)'''

    # confusion matrix 
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.title(f"Confusion matrix for {clf.__class__.__name__} \n\nData size: {len(y_test)} test data / {len(y_train)} training data")
    plt.show()
    print(classification_report(y_test, predictions, digits=4))



def train_bert(X_train, X_test, X_val, y_train, y_test, y_val):

    # (1) settings
    batch_size = 16 # number of texts to process at once
    init_lr = 2e-5 # initial learning rate
    end_lr = 0 # end learning rate
    warmup_proportion = 0.1 # proportion of training steps to warm up the learning rate
    num_epochs = 5 # number of training epochs

    max_seq_length = 128 # maximum sequence length for the input, can be increased if needed

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    # (2) load and prepare dataset
    # turn text into tokens
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=max_seq_length, return_tensors="tf")
    val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True, max_length=max_seq_length, return_tensors="tf")
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=max_seq_length, return_tensors="tf")

    # create tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train
    )).shuffle(1000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        y_val
    )).shuffle(1000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test
    )).batch(batch_size)

    # print dataset info
    print(f"Dataset created.\n\nThe training dataset  has {len(train_dataset)} samples, the test dataset has {len(test_dataset)} samples and the validation dataset has {len(val_dataset)}.")
    steps = math.ceil(len(X_train) / batch_size)
    num_warmup_steps = round(steps * warmup_proportion * num_epochs)
    print(f'You are planning to train for a total of {steps} steps * {num_epochs} epochs = {num_epochs*steps} steps. Warmup is {num_warmup_steps}, {round(100*num_warmup_steps/(steps*num_epochs))}%. We recommend at least 10%.')


    # (3) train the model
    train_steps_per_epoch = int(len(train_dataset) / batch_size)
    num_train_steps = train_steps_per_epoch * num_epochs

    # initalize a model for sequence classification with two labels
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=init_lr, decay_steps=num_train_steps, end_learning_rate=end_lr)
    # using legacy because of M1 chip
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
   
    # compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # start training
    model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs, batch_size=batch_size)
    
    print(f'The training has finished training after {num_epochs} epochs.')

    # save model
    model.save_weights("mymodel.h5")

    # load model
    model = model.load_weights("mymodel.h5")

    print("Evaluating test set")
    y_pred = model.predict(model.test_dataset)
    y_pred_bool = y_pred["logits"].argmax(-1)
    print(classification_report(y_test, y_pred_bool, digits=4))