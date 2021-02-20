# import packages
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from nltk.stem import WordNetLemmatizer

import re 
from nltk import word_tokenize 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import joblib

def load_data(data_file):
    # read in file

    engine = create_engine('sqlite:///disaster.db')
    df = pd.read_sql_table('disaster', con = engine)

    # clean data
    
    # load to database

    # define features and label arrays

    X = df.iloc[:,1].values
    y = df.iloc[:,5:-1].values

    return X, y

def tokenize(text):
    # sentence tokenize 
    sentences = sent_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for sent_ in sentences:
        # normalization 
        text = re.sub(r"[^a-zA-Z0-9]", " ", sent_.lower())
        # tokenize 
        words = word_tokenize(text)
        # remove stop words 
        words = [word for word in words if not word in stop_words]
        # lemmatization
        for word in words:
            clean_tok = lemmatizer.lemmatize(word).lower().strip()
            clean_tokens.append(clean_tok)
    return clean_tokens    


def build_model():
    # text processing and model pipeline

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),      
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define parameters for GridSearchCV

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }

    # create gridsearch object and return as final model pipeline

    cv_model = GridSearchCV(pipeline, param_grid = parameters)

    return cv_model


def train(X, y, model):
    # train test split

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.3, random_state = 0)

    # fit model
    model.fit(X_train, y_train)

    # output model test results
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model


def export_model(model):
    # Export model as a pickle file
    filename = 'final_model.pkl'
    joblib.dump(model, filename)


def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    # data_file = sys.argv[1]  # get filename of dataset
    data_file = 'disaster.db'  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline