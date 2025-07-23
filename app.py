from flask import Flask,render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare the movie dataset (like in your backend code)
df = pd.read_csv('imdb.csv')

df.rename(columns={'Rating_from_10': 'Rating'}, inplace=True)


# Fill or drop NaN values in critical columns (Metascore, Gross_in_$_M, Rating)
df['Metascore'].fillna(df['Metascore'].mean(), inplace=True)  # Fill NaNs with the mean value

# Clean 'Gross_in_$_M' column: remove non-numeric characters and convert to float
df['Gross_in_$_M'] = df['Gross_in_$_M'].replace(r'[^\d.]', '', regex=True).astype(float)
df['Gross_in_$_M'].fillna(df['Gross_in_$_M'].mean(), inplace=True)  # Fill NaNs with the mean value

genre_df = df['Genre'].str.get_dummies(sep=',')
df = pd.concat([df, genre_df], axis=1)

# Prepare the content features and normalize the data
content_features = ['Metascore', 'Gross_in_$_M', 'Rating'] + genre_df.columns.tolist() # type: ignore
content_matrix = df[content_features]

# Check for NaN in the content matrix before normalization
print(content_matrix.isnull().sum())  # Check if any NaN values remain

# Normalize the content matrix
scaler = StandardScaler()
normalized_content_matrix = scaler.fit_transform(content_matrix[['Metascore', 'Gross_in_$_M', 'Rating']])


# Compute the cosine similarity
movie_similarity = cosine_similarity(normalized_content_matrix)
similarity_df = pd.DataFrame(movie_similarity, index=df['Movie_name'], columns=df['Movie_name'])

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html') 

# Route to handle movie recommendation
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    movie_name = request.json.get('movie_name')

    if not movie_name:
        return jsonify({'error': 'No movie name provided'}), 400

    if movie_name not in similarity_df.columns:
        return jsonify({'error': 'Movie not found in the database'}), 404

    similar_movies = similarity_df[movie_name].sort_values(ascending=False).iloc[1:6]
    recommendations = similar_movies.index.tolist()

    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
