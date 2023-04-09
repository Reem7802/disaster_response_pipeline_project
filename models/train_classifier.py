import sys
import re
import numpy as np
import nltk
import pickle
import pandas as pd

nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine

from sklearn.metrics import  classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):

    '''
    This function load and read the data from sqlite database

    Input:

    database_filepath

    Output:
    
    X, Y, Y_columns
    
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    columns = df.columns
    Y_columns = columns[4:]
    X = df['message']
    Y = df[Y_columns]
    Y = Y.astype(bool)
    return X, Y, Y_columns

   
def tokenize(text):

    '''
    This function tokenize the text 

    Input:

    text

    Output:
    
    clean text
    
    '''
    text = text.lower()
    text = re.sub(r'[^A-Za-z]', ' ', text)
    tokens = word_tokenize(text)

    lemmatizer =WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():

    '''
    This function build the model to classifie text to categories 

    Input:

    Output:
    
    model
    
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2],
    
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)
    return model


def evaluate_model(model, X_test, y_test, category_names):

    '''
    This evaluate model by printing a full report shows  precision, recall, f1-score, support 

    Input:

    model, X_test, y_test, category_names

    Output:
    
    report
    
    '''
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=category_names)
    print(report)

def save_model(model, model_filepath):

    '''
    This evaluate saves model after training in pkl file

    Input:

    model, model_filepath

    Output:
    
    classifier.pkl
    
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()