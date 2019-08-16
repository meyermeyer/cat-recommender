from flask import Flask, jsonify, request
from surprise import Reader, Dataset

import pandas as pd


app = Flask(__name__)


print('surprise test')

# data frame read the csv, data is separated by commas
ratings = pd.read_csv('./data/ratings.csv', sep=',')
ratings.head()

ratings_dict = {
    'itemID': list(ratings.movieId),
    'userID': list(ratings.userId),
    'rating': list(ratings.rating)
}

df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale=(0.5,5.0))

data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']],reader)

print(data)


if __name__ == '__main__':
    app.run(debug=True)
