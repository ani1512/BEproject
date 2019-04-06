#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:32:36 2019

@author: anish
"""
import pandas as pd

#getting the data
dataset = pd.read_csv('movie_metadata.csv',
                      usecols=['movie_title','genres','actor_1_name',
                               'plot_keywords','director_name'])

corpus = [] #will contain the bag of words

#cleaning the data and creating bag of words
import re
for i in range (0,len(dataset.index)) :
    director_name = re.sub('[^a-zA-Z]', 
                           '', dataset['director_name'][i]).lower()
    actor = re.sub('[^a-zA-Z]', '', dataset['actor_1_name'][i]).lower()
    genres = re.sub('[|]', ' ', dataset['genres'][i]).lower()
    plot_keywords = re.sub('[ ]', '', dataset['plot_keywords'][i])
    plot_keywords = re.sub('[|]', ' ', plot_keywords)
    corpus.append(' '.join((director_name, actor, genres, plot_keywords)))
    
#Building the sparse matrix out of the bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 12000)
X = cv.fit_transform(corpus).toarray()

#generating the similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(X, X)

"""function to generate top 10 similar movies from a given movie using the
similarity matrix"""
def recommendations(title, cosine_sim = cosine_sim):
    
    #initializing empty list of recommended movies
    recommended_movies = []
    
    #getting the index of the movie that matches the title
    index = dataset.loc[dataset['movie_title'] == title].index[0]
    
    #create a series with similarity scores in descending order
    score_series = pd.Series(cosine_sim[index]).sort_values(ascending = False)
    
    #getting the indices of the 10 most similar movies
    top_10_indices = list(score_series.iloc[1:11].index)
    
    #populating with top 10 similar movie titles
    for i in top_10_indices:
        recommended_movies.append(dataset['movie_title'][i])
    
    return recommended_movies

rm = recommendations('Thor\xa0')