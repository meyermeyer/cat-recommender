import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import pandas as pd
import numpy as mp
import warnings
warnings.filterwarnings('ignore')

# app.debug = True

app = Flask(__name__)
api = Api(app)
print('test')


class Movies(Resource):
    def get(self):
        
        return 'test'


# @app.route('/greet')
# def say_hello():
#   return 'Hello from Server'

# @app.route('/movie', methods=['GET'])

# def movieRoutes():
#     if request.method == 'GET':
#         print(request.query)
#         return 200

# data frame read the csv, data is separated by commas
df = pd.read_csv('./data/ratings.csv', sep=',')
# shows first 5 rows 
df.head()
# read movies.csv
movie_titles = pd.read_csv('./data/movies.csv', sep=',')
movie_titles.head()
# ratings.csv and movies.csv
df = pd.merge(df, movie_titles, on='movieId')
# print(df.head())
# show stats from dataset
# print(df.describe())
# return DataFrame of average rating for each title
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
# print(ratings.head())
# create a column to display number of ratings
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
# print(ratings.head)

# load histogram of ratings
plt.hist(ratings['rating'], bins=50)
# histo of number of ratings
plt.hist(ratings['number_of_ratings'], bins=60)
# use seabord to scatter plot number of ratings and rating
sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
# plt.show()

# create a matrix with movie titles as columns, userId as indexes, and ratings as values
movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
# print(movie_matrix.head())

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

api.add_resource(Movies, '/movies')

if __name__ == '__main__':
     app.run(port='5002')
