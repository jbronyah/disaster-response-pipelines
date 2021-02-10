'''
DESCRIPTION

This module takes the cleaned data from the ETL module and trains a ML classification model
The output is a model which can be used to predict on new data

INPUTS

database_filepath -   path containing the processed data database


OUTPUTS

Saves the trained model to a pickle file

SCRIPT EXECUTION SAMPLE

python train_classifier.py ../data/Disaster_Response_Data.db classifier.pkl


'''
# import relevant libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    
    '''
    DESCRIPTION
    Loads processed data from database at given file path
    
    INPUTS
    database_filepath - path containing the processed data database
    
    OUTPUTS
    X - dataframe of features for prediction model
    Y - Series of field we want to predict
    category_names - names of each column in dataset
    
    '''
    # load data from database
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)
    
    # from inspection of data we see the 'related' column has some values set to 2. we impute those values to 1 since it's most dominant
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    # get X,y values to be used for classification
    
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    # get column category list
    category_names = list(np.array(Y.columns))
    
    return X, Y, category_names


def tokenize(text):
    '''
    DESCRIPTION
    Normalizes and tokenizes text input given to it
    URLs and email addresses are also replaced with generic placeholders and tokens are lemmatized
    
    INPUTS
    text - sample text data from X dataframe
    
    OUTPUTS
    tokens_list - list of tokenized words from given input text
    
    '''
    # regex to identify URLs and email addresses
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    email_regex = '^[\w\.\+\-]+\@[\w]+\.[a-z]{2,3}$'
    
    # find and store all URLs and email addresses
    
    urls_found = re.findall(url_regex, text)
    emails_found = re.findall(email_regex, text)
   
    # replace all found URLs and email addresses with placeholder names
    
    for url in urls_found:
        text = text.replace(url, "urlplaceholder")
    
    for email in emails_found:
        text = text.replace(email, "emailplaceholder")
    
    # tokenize text input
    
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    tokens_list = []
    
    # lemmatize text and convert text to lowercase
    
    for word in words:
        word_token = lemmatizer.lemmatize(word).lower().strip()
        tokens_list.append(word_token)
    
    return tokens_list


# Custom Transformer class to  get starting verb of text

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)



def build_model():
    
    '''
    DESCRIPTION
    Creates a pipeline for building and training classification model
    
    OUTPUTS
    cv - best model output from Gridsearch operation on the pipeline
    
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transform', TfidfTransformer())
            ])),
            
             ('starting_verb', StartingVerbExtractor())

        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {'classifier__estimator__learning_rate': [0.01, 0.1],
                   'classifier__estimator__n_estimators': [16, 32]}
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    DESCRIPTION
    Evaluates and prints scores of model on test set data
    
    INPUTS
    model - classification model
    X_test - test set for X
    Y_test - test set for Y
    category_names - names of each column in dataset
    
    '''
    y_pred = model.predict(X_test)
    
    print(classification_report(Y_test.values, y_pred, target_names=category_names))
    
    


def save_model(model, model_filepath):
    
    '''
    DESCRIPTION
    Saves classfication model for later use
    
    INPUTS
    model - classification model
    file_path - specified name of model
    
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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