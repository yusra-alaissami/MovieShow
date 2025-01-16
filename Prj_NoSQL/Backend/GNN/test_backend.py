# test_backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from data_loader import load_data
from train_gnn import train_gnn, GNN
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load saved embeddings

def load_saved_data(file_path="node_embeddings.pth"):
    try:
        saved_data = torch.load(file_path)
        return saved_data['embeddings'], saved_data['node_mapping']
    except FileNotFoundError:
        return None, None

# Recommend movies
def recommend_movies(movie_title, embeddings, node_mapping, k=10):
    index_to_title = {idx: title for title, idx in node_mapping.items()}
    movie_idx = node_mapping.get(movie_title)

    if movie_idx is None:
        return []

    movie_embedding = embeddings[movie_idx].unsqueeze(0)
    similarities = cosine_similarity(movie_embedding, embeddings)[0]
    similar_indices = similarities.argsort()[-k-1:-1][::-1]
    recommendations = [index_to_title[idx] for idx in similar_indices if idx in index_to_title]
    return recommendations

# Train route
@app.route("/train", methods=["POST"])
def train():
    try:
        data = load_data()
        train_gnn(data)
        return jsonify({"message": "Model trained and embeddings saved successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Recommend route
@app.route("/recommend", methods=["GET"])
def recommend():
    movie_title = request.args.get("title")
    embeddings, node_mapping = load_saved_data()
    if embeddings is None or node_mapping is None:
        return jsonify({"error": "Model data not available"}), 500

    recommendations = recommend_movies(movie_title, embeddings, node_mapping)
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True, port=5008)


