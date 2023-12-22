from flask import Flask, render_template, request
import pickle
from tmdbv3api import TMDb, Movie

app = Flask(__name__)

try:
    movies = pickle.load(open('movies.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'movies.pkl' file not found. Please check the file path.")
    exit()

# using bag of words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
cv = CountVectorizer(max_features=5000, stop_words='english')

# converts each sentence into vectors
vectors = cv.fit_transform(movies['tag']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Configure TMDb API
tmdb = TMDb()
tmdb.api_key = 'c23ea6dbf1ea9800335aba342a81139e'  # API key

def get_poster_path(movie_id):
    try:
        movie = Movie()
        details = movie.details(movie_id)
        return details.poster_path
    except Exception as e:
        print(f"Error fetching poster path for movie ID {movie_id}: {e}")
        return None

def recommend(movie):
    matching_movies = movies[movies['title'] == movie]
    
    if matching_movies.empty:
        print(f"No movies found with the title '{movie}'.")
        return []

    movie_index = matching_movies.index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        movie_info = movies.iloc[i[0]]
        poster_path = get_poster_path(movie_info['movie_id'])
        recommended_movies.append({'title': movie_info['title'], 'poster_path': poster_path})

    print(f"Recommendations: {recommended_movies}")
    return recommended_movies

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    user_input = request.form['movie_title']
    recommendations = recommend(user_input)
    return render_template('index.html', input_movie=user_input, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

