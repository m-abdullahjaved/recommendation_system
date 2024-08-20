# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv2').output)

# Function to preprocess images and extract features using the CNN model
def extract_cnn_features(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    features = features.flatten()  # Flatten to create a feature vector
    return features

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with open('model.pickle', 'rb') as file:
    reduced_features = pickle.load(file)

with open('1M-features.pickle', 'rb') as file:
    features = pickle.load(file)


movie_columns = ["MovieID", "Title", "Genres"]
ratings_columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
# Specify the delimiter in the movies.dat file
delimiter = "::"


movies_file_path = "static/movies.dat"
ratings_file_path = "static/ratings.dat"
rec_file_path = "static/final_prediction_1m.csv"

#movies_df = pd.read_csv(movies_file_path)
movies_df = pd.read_csv(movies_file_path,  encoding="ISO-8859-1",  sep=delimiter, header=None, names=movie_columns)
ratings_df = pd.read_csv(ratings_file_path,  encoding="ISO-8859-1", sep=delimiter, header=None, names=ratings_columns)
rec_df = pd.read_csv(rec_file_path, header = 0 ,names = ['UserID', 'MovieID', 'Actual Rating', 'Predicted'])


def search_movie(title):
    api_key = '4d4df698'
    url = f'http://www.omdbapi.com/?apikey={api_key}&t={title}'
    response = requests.get(url)
    data = response.json()
    return data


@app.route('/')
def similar_movie():
    return render_template('similar_movie.html')


@app.route('/similar_poster')
def similar_poster():
    return render_template('similar_poster.html')


@app.route('/collaborative')
def collaborative():
    return render_template('collaborative.html')


@app.route('/hybrid')
def hybrid():
    return render_template('hybrid.html')


@app.route('/details')
def details():
    selected_movie = request.args.get('selected_movie')
    if selected_movie:
        selected_movie.replace('%20', ' ')
        selected_movie = selected_movie[:len(selected_movie) - 6]

    movie1_details = search_movie(selected_movie)


    return render_template('collaborative.html', movie1_details=movie1_details)


@app.route('/recommendPoster', methods=['POST'])
def recommendPoster():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        recommendations = recommend_similar_movies(filename, features, movies_df)
        return render_template('similar_poster.html', filepath=filepath, recommendations=recommendations)


@app.route('/similarityPoster')
def similarityPoster():
    clicked_movie = request.args.get('clicked_movie')

    if clicked_movie:
        clicked_movie.replace('%20', ' ')
        clicked_movie = clicked_movie[:len(clicked_movie) - 6]

    movie2_details = search_movie(clicked_movie)

    return render_template('similar_poster.html', movie2_details=movie2_details)


@app.route('/similarity')
def similarity():

    selected_movie = request.args.get('selected_movie')
    clicked_movie = request.args.get('clicked_movie')
    if selected_movie:
        selected_movie.replace('%20', ' ')
        selected_movie = selected_movie[:len(selected_movie) - 6]

    movie1_details = search_movie(selected_movie)
    if clicked_movie:
        clicked_movie.replace('%20', ' ')
        clicked_movie = clicked_movie[:len(clicked_movie) - 6]

    movie2_details = search_movie(clicked_movie)

    return render_template('similar_movie.html', movie1_details=movie1_details, movie2_details=movie2_details)


def recommend_similar_movies(target_poster, reduced_features, movies_df):
    # Retrieve the features of the target poster
    target_features = reduced_features.get(target_poster)
    if target_features is None:
        target_features = extract_cnn_features(os.path.join(app.config['UPLOAD_FOLDER'], target_poster))

    similarity_scores = {}

    for poster, features in reduced_features.items():
        similarity = cosine_similarity([target_features], [features])[0][0]
        similarity_scores[poster] = similarity

    # Sort movies by similarity scores in descending order
    recommended_movies = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    # Extract movie titles based on movie IDs
    recommended_movies_with_titles = []
    for movie_id, similarity_score in recommended_movies:
        movie_id_numeric = int(movie_id.split('.')[0])
        movie_title = movies_df[movies_df['MovieID'] == movie_id_numeric]['Title'].iloc[0]
        movie_genre = movies_df[movies_df['MovieID'] == movie_id_numeric]['Genres'].iloc[0]
        recommended_movies_with_titles.append((movie_id, movie_title, similarity_score, movie_genre))

    return recommended_movies_with_titles


@app.route('/previous_rated', methods=['POST'])
def history():
    if request.method == 'POST':
        previously_rated = []
        userId = int(request.form.get('user_id'))
        previously_rated = ratings_df[ratings_df['UserID'] == userId]
        previously_rated.merge(movies_df, on='MovieID')
        previously_rated = pd.merge(previously_rated, movies_df, on='MovieID')
        previously_rated = previously_rated[['MovieID', 'Title', 'Rating']]
        previously_rated = list(previously_rated.to_records(index=False))
        #print(previously_rated)
        return render_template('collaborative.html', previously_rated=previously_rated)
    return ('collaborative.html')


@app.route('/hybridRec', methods=['POST'])
def hybridRec():
    if request.method == 'POST':
        recommendations = []
        #print(rec_df)
        userId = int(request.form.get('user_id'))
        recommendations = rec_df[rec_df['UserID'] == userId]
        recommendations.merge(movies_df, on='MovieID')
        recommendations = pd.merge(recommendations, movies_df, on='MovieID')
        recommendations = recommendations[['MovieID', 'Title', 'Actual Rating', 'Predicted']]
        recommendations = list(recommendations.to_records(index=False))
        #print(recommendations)
        return render_template('hybrid.html', recommendations=recommendations)
    return ('hybrid.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        movie_name = request.form.get('movie_name')  # Assuming the input field name is 'movie_name'
        #print(movie_name)

        # Search for similar movie in the movies_df DataFrame
        similar_movie = movies_df[movies_df['Title'].str.contains(movie_name, case=False)]
        if not similar_movie.empty:
            similar_movie_id = similar_movie.iloc[0]['MovieID']
            similar_movie_id = str(similar_movie_id) + ".jpg"
            recommendations = recommend_similar_movies(similar_movie_id, reduced_features, movies_df)
            #print(recommendations)
            return render_template('similar_movie.html', recommendations=recommendations)
        else:
            return "Movie not found!"

    return render_template('similar_movie.html')  # Render the template with no recommendations if it's not a POST request


if __name__ == '__main__':
    app.run(debug=True)