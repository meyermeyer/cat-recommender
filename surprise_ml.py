from flask import Flask, jsonify, request
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD, evaluate
from surprise import NMF
from collections import defaultdict
import pandas as pd


app = Flask(__name__)


print('surprise test')

# data frame read the csv, data is separated by commas
df = pd.read_csv('./data/ratings.csv', sep=',')

# read movies.csv
movie_titles = pd.read_csv('./data/movies.csv', sep=',')

# ratings.csv and movies.csv
df = pd.merge(df, movie_titles, on='movieId')

# return DataFrame of average rating for each title
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

# create a column to display number of ratings
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()

# create a matrix with movie titles as columns, userId as indexes, and ratings as values
movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
# movie_matrix = movie_matrix.fillna()
print(movie_matrix.head())




data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

testset = trainset.build_anti_testset()
predictions = algo.test(testset)
all_predictions = {}
all_predictions['uid'] = {}
all_predictions['uid']['iid'] = 'est'
# each_rating = {'iid' : 'est'}
# prediction_by_uid = defaultdict(list)
# all_predictions = {'uid' : {'iid' : 'est'}}
for uid, iid, true_r, est, _ in predictions:
    all_predictions[uid] = {} 
    all_predictions[uid][iid] = est
    # all_predictions[uid] = 


print(all_predictions)

# NMF - non negative matrix factorization
algo = NMF()
# evaluate(algo, data, measures=['RMSE'])


if __name__ == '__main__':
    app.run(debug=True)
