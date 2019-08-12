from flask import Flask, jsonify, request
import pandas as pd
import numpy as mp
import warnings
warnings.filterwarnings('ignore')

# app.debug = True

app = Flask(__name__)
print('test')
df = pd.read_csv('./data/ratings.csv', sep=',', names=['user_id', 'item_id', 'rating', 'timestamp'])
print(df.head())
