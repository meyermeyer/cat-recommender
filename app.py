from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD, evaluate
from surprise import NMF
from collections import defaultdict
import warnings

import surprise_ml
from surprise_ml import svd_movie_matrix
from surprise_ml import nmf_movie_matrix

warnings.filterwarnings('ignore')

# app.debug = True

app = Flask(__name__)
api = Api(app)

# print('test')

# data frame read the csv, data is separated by commas
df = pd.read_csv('./data/ratings.csv', sep=',')
# shows first 5 rows
df.head()

# read movies.csv
movie_titles = pd.read_csv('./data/movies.csv', sep=',')
# print(movie_titles.head())
# ratings.csv and movies.csv
df = pd.merge(df, movie_titles, on='movieId')

# show stats from dataset
# print(df.describe())

# return DataFrame of average rating for each title
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

# create a column to display number of ratings
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
print(ratings['number_of_ratings'])

# create a matrix with movie titles as columns, userId as indexes, and ratings as values
movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
# print(movie_matrix)

class Recommendations(Resource):
    def get(self, title):
        # all ratings for selected title
        title_user_rating = movie_matrix[title]
        # print(title_user_rating)
        # correlation to title
        similar_to_title = movie_matrix.corrwith(title_user_rating)
        print(similar_to_title)
        # drop null values from matrix and transform correlation results into DataFrame
        corr_title = pd.DataFrame(
            data=similar_to_title, columns=['Correlation'])
        corr_title.dropna(inplace=True)
        # join correlations with number of ratings column in the ratings datafield
        corr_title = corr_title.join(ratings['number_of_ratings'])
        # set a threshold for number of ratings
        top_eleven = corr_title[corr_title['number_of_ratings']>100].sort_values(by='Correlation', ascending=False).head(11)
        #  use to_dict to convert to dictionary, exclude 'title' by requiring correlation less than .99
        json_object = top_eleven[top_eleven['Correlation']<.99].to_dict('index')
        print(json_object)
        if json_object == {} :
            return {'recommendations' : 'none'}
        else :
            return {'recommendations': json_object}
        # return {'hello': 'world'}


api.add_resource(Recommendations, '/movies/recommendations/<string:title>')


class SvdRecommendations(Resource):
    def get(self, title):
        print('title', title)
        iid = movie_titles.loc[movie_titles['title']
                               == title, 'movieId'].item()
        print(iid)

        # all ratings for selected title
        title_user_rating = svd_movie_matrix[iid]
        print('title_user_rating', title_user_rating)
        # correlation to title
        similar_to_title = svd_movie_matrix.corrwith(title_user_rating)
        print(similar_to_title)
        # drop null values from matrix and transform correlation results into DataFrame
        corr_title = pd.DataFrame(
            data=similar_to_title, columns=['Correlation'])

        print('corr_title', corr_title.sort_values(
            by='Correlation', ascending=False))

        # set a threshold for number of ratings
        top_eleven = corr_title.sort_values(
            by='Correlation', ascending=False).head(11)

        top_eleven = pd.merge(top_eleven, movie_titles, on='movieId')
        print('top_eleven', top_eleven)

        #  use to_dict to convert to dictionary, exclude 'title' by requiring correlation less than .99
        json_object = top_eleven[top_eleven['Correlation'] < .99].to_dict(
            'index')

        print(json_object)
        if json_object == {}:
            return {'recommendations': 'none'}
        else:
            return {'recommendations': json_object}


api.add_resource(SvdRecommendations,
                 '/movies/recommendations/SVD/<string:title>')

class NmfRecommendations(Resource):
    def get(self, title):
        print('title', title)
        iid = movie_titles.loc[movie_titles['title']
                               == title, 'movieId'].item()
        print(iid)

        # all ratings for selected title
        title_user_rating = nmf_movie_matrix[iid]
        print('title_user_rating',title_user_rating)
        # correlation to title
        similar_to_title = nmf_movie_matrix.corrwith(title_user_rating)
        print(similar_to_title)
        # drop null values from matrix and transform correlation results into DataFrame
        corr_title = pd.DataFrame(
            data=similar_to_title, columns=['Correlation'])
        
        print('corr_title', corr_title.sort_values(by='Correlation', ascending=False))

        # set a threshold for number of ratings
        top_eleven = corr_title.sort_values(
            by='Correlation', ascending=False).head(11)
        
        top_eleven = pd.merge(top_eleven, movie_titles, on='movieId')
        print('top_eleven', top_eleven)
       
       
        #  use to_dict to convert to dictionary, exclude 'title' by requiring correlation less than .99
        json_object = top_eleven[top_eleven['Correlation'] < .99].to_dict(
            'index')

        print(json_object)
        if json_object == {}:
            return {'recommendations': 'none'}
        else:
            return {'recommendations': json_object}

api.add_resource(NmfRecommendations, '/movies/recommendations/NMF/<string:title>')


class Movies(Resource):
    def get(self):
        print('in Movies route')
        movie_titles = list(pd.read_csv('./data/movies.csv', sep=',')['title'])
        # print(movie_titles)
        return jsonify({'movie_titles': movie_titles})
        # return{'movie_titles': movie_titles}
        

api.add_resource(Movies, '/movies')




# load histogram of ratings
# plt.hist(ratings['rating'], bins=50)
# histo of number of ratings
# plt.hist(ratings['number_of_ratings'], bins=60)
# use seabord to scatter plot number of ratings and rating
# sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
# plt.show()




# show the ten movies with the most ratings
# print(ratings.sort_values('number_of_ratings', ascending=False).head(10))

# show all user ratings for Air Force One and Contact
AFO_user_rating = movie_matrix['Air Force One (1997)']
contact_user_rating = movie_matrix['Contact (1997)']
# print(AFO_user_rating.head(10), contact_user_rating.head(10))

# correlation to AFO
similar_to_AFO = movie_matrix.corrwith(AFO_user_rating)
# print(similar_to_AFO.head())

# correlation to contact
similar_to_contact = movie_matrix.corrwith(contact_user_rating)
# print(similar_to_contact.head())

# drop null values from matrix and transform correlation results into DataFrames
corr_contact = pd.DataFrame(similar_to_contact, columns = ['Correlation'])
corr_contact.dropna(inplace=True)
# print(corr_contact.head())

corr_AFO = pd.DataFrame(similar_to_AFO, columns = ['Correlation'])
corr_AFO.dropna(inplace=True)
# print(corr_AFO.sort_values('Correlation', ascending=False).head())



# join correlations with number of ratings column in the ratings datafield
corr_AFO = corr_AFO.join(ratings['number_of_ratings'])
corr_contact = corr_contact.join(ratings['number_of_ratings'])
# print(corr_AFO.head(), corr_contact.head())

# set a threshold for number of ratings
# print(corr_AFO[corr_AFO['number_of_ratings']>100].sort_values(by='Correlation', ascending=False).head(10))

# print(corr_contact[corr_contact['number_of_ratings']>100].sort_values(by='Correlation', ascending=False).head(10))

if __name__ == '__main__':
    app.run(debug=True)
