from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD, evaluate
from surprise import NMF
from collections import defaultdict
import pandas as pd
import numpy as np
import sys
from fastai.datasets import *

print(sys.path)

app = Flask(__name__)

print('fast_ai')

ratings = pd.read_csv('./data/ratings.csv', sep=',')
print(ratings.head())

x = ratings.drop(['rating'], axis=1)
y = ratings['rating'].astype(np.float32)
# data = ColumnarModelData.from_data_frame(
#     path, val_indx, x, y, ['userId', 'movieId'], 64)
# print(data.head())

if __name__ == '__main__':
    app.run(debug=True)
