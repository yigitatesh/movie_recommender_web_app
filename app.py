# Import Libraries
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.neighbors import KNeighborsClassifier

from flask import Flask, request, jsonify, render_template, flash
import os

# Create Flask app
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# LOAD DATA
X = load_npz("data/X_data.npz")
y = pd.read_csv("data/y_data.csv")

# CREATE MODEL
knn = KNeighborsClassifier().fit(X, y.values.ravel())


# HELPER FUNCTIONS

def get_similar_movies(movie_name, X, y, knn, n_movies=10):
    """Returns n similar to the given movie"""
    index = y.loc[y["title"].str.lower() == movie_name].index[0]
    movie_data = X[index, :].toarray()
    distances, indices = knn.kneighbors(movie_data, n_neighbors=n_movies+1)
    
    movies = []
    for i in np.squeeze(indices):
        movie = y.iloc[i]["title"]
        movies.append(movie)
    
    return movies[1:]

def process_count(count):
    count = count.strip()
    if count == "":
        return 1
    elif count.isdigit():
        count = int(count)
        if count < 1:
            return 1
        return count
    else:
        return None


# Flask Functions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend():
    '''
    For rendering results on HTML GUI
    '''
    # get movie name
    ## preprocess given features
    features = list(request.form.values())
    
    # get movie name
    movie_name = features[0].lower()
    if movie_name.strip() == "":
        flash("Please type a movie name.")
        return render_template("index.html", condition=-2) # error

    # get movie count
    n_movies = process_count(features[1])
    if n_movies is None:
        flash("Please type a number as number of movies to recommend.")
        return render_template("index.html", condition=-2) # error
    
    # name is in dataset
    if movie_name in y["title"].str.lower().values:
        condition = 1 # movie found
        movies = get_similar_movies(movie_name, X, y, knn, n_movies=n_movies)
    
    # some movie names contains searched movie name
    elif y["title"].str.lower().str.contains(movie_name).any():
        condition = 0 # similar named movies found
        movies = y[y["title"].str.lower().str.contains(movie_name)].values.ravel()
    
    # not founded at all
    else:
        condition = -1 # no similar name found
        movies = []

    return render_template('index.html', movies=movies, condition=condition)


if __name__ == "__main__":
    app.run(debug=True)