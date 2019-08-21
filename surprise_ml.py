from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD, evaluate
from surprise import NMF
from collections import defaultdict
import pandas as pd

app = Flask(__name__)
api = Api(app)

print('surprise test')

# data frame read the csv, data is separated by commas
df = pd.read_csv('./data/ratings.csv', sep=',')

# read movies.csv
movie_titles = pd.read_csv('./data/movies.csv', sep=',')
print(movie_titles.head())

# ratings.csv and movies.csv
df = pd.merge(df, movie_titles, on='movieId')

# return DataFrame of average rating for each title
ratings = pd.DataFrame(df.groupby('movieId')['rating'].mean())
print(ratings.head())

# create a column to display number of ratings
ratings['number_of_ratings'] = df.groupby('movieId')['rating'].count()

# create a matrix with movie titles as columns, userId as indexes, and ratings as values
movie_matrix = df.pivot_table(
    index='userId', columns='movieId', values='rating')

data = Dataset.load_builtin('ml-100k')
print(data)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

testset = trainset.build_anti_testset()
predictions = algo.test(testset)

all_predictions = {}
all_predictions['uid'] = {}
all_predictions['uid']['iid'] = 'est'

for uid, iid, true_r, est, _ in predictions:
    all_predictions[uid] = {} 
    all_predictions[uid][iid] = est
    # all_predictions[uid] = 

svd_movie_matrix = movie_matrix.fillna(all_predictions[uid][iid])


class Recommendations(Resource):
    def get(self, title):
        print('title', title)
        iid = movie_titles.loc[movie_titles['title'] == title, 'movieId'].item()
        print(iid)
    
        # all ratings for selected title
        title_user_rating = svd_movie_matrix[iid]
        # print(title_user_rating)
        # correlation to title
        similar_to_title = svd_movie_matrix.corrwith(title_user_rating)
        print(similar_to_title)
        # drop null values from matrix and transform correlation results into DataFrame
        corr_title = pd.DataFrame(
            data=similar_to_title, columns=['Correlation'])
        corr_title.dropna(inplace=True)
        # join correlations with number of ratings column in the ratings datafield
        corr_title = corr_title.join(ratings['number_of_ratings'])
        # set a threshold for number of ratings
        top_eleven = corr_title[corr_title['number_of_ratings'] > 100].sort_values(
            by='Correlation', ascending=False).head(11)
        #  use to_dict to convert to dictionary, exclude 'title' by requiring correlation less than .99
        json_object = top_eleven[top_eleven['Correlation'] < .99].to_dict(
            'index')
        print(json_object)
        if json_object == {}:
            return {'recommendations': 'none'}
        else:
            return {'recommendations': json_object}

        # return {'hello': 'world'}

api.add_resource(Recommendations, '/movies/recommendations/SVD/<string:title>')

# NMF - non negative matrix factorization
algo = NMF()
# evaluate(algo, data, measures=['RMSE'])


if __name__ == '__main__':
    app.run(debug=True)
