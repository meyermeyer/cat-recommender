from flask import Flask, jsonify, request
import pandas as pd
import numpy as mp
import warnings
warnings.filterwarnings('ignore')

# app.debug = True

app = Flask(__name__)
print('test')
df = pd.read_csv('./data/ratings.csv', sep=',')
print(df.head())
movie_titles = pd.read_csv('./data/movies.csv', sep=',')
print(movie_titles.head())
df = pd.merge(df, movie_titles, on='movieId')
print(df.head())
