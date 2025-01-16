from flask import Flask, request, jsonify
from test import load_saved_data, recommend_movies

from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)

@app.route('/recommend', methods=['GET'])
def recommend():
    movie_title = request.args.get('movie', default='', type=str)
    embeddings, node_mapping = load_saved_data()
    if embeddings is None or node_mapping is None:
        return jsonify([])

    recommendations = recommend_movies(movie_title, embeddings, node_mapping, k=5)
    return jsonify([{'title': title, 'image': 'https://via.placeholder.com/150'} for title in recommendations])

if __name__ == "__main__":
    app.run(debug=True, port=5001)

