import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

# importing libraries for custom transformer
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

# adding custom transformer class

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

    
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/Disaster_Response_Data.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # get category by count
    cat_counts = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum().sort_values(ascending=False)
    cat_names = list(cat_counts.index)
    
    # get category and genre percentages
    genre_perc = round(100*genre_counts/genre_counts.sum(), 1)
    cat_perc = round(100*cat_counts/cat_counts.sum(), 1)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            "data": [
              {
                "type": "pie",
                "hole": 0.2,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": genre_perc,
                  "y": genre_names
                },
                "marker": {
                  "colors": [
                    "#8b0000",
                    "#006400",
                    "#000080"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": genre_names,
                "values": genre_perc
              }
            ],
            "layout": {
              "title": "Genre Percentage Distribution"
            }
        },
        
        {
            "data": [
              {
                "type": "bar",
                "x": cat_names,
                "y": cat_perc,
                "marker": {
                  "color": 'blue'}
                }
            ],
            "layout": {
              "title": "Message Distribution by Category",
              'yaxis': {
                  'title': "Percentage(%)"
              },
              'xaxis': {
                  'title': "Category"
              },
              'barmode': 'group'
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()