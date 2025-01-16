import torch
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_data
from train_gnn import train_gnn

def load_saved_data(file_path="node_embeddings.pth"):
    try:
        saved_data = torch.load(file_path)
        return saved_data['embeddings'], saved_data['node_mapping']
    except FileNotFoundError:
        print("Error: Embedding file not found.")
        return None, None

def recommend_movies(movie_title, embeddings, node_mapping, k=5):
    index_to_title = {idx: title for title, idx in node_mapping.items()}
    movie_idx = node_mapping.get(movie_title)

    if movie_idx is None:
        print(f"Movie '{movie_title}' not found.")
        return []

    movie_embedding = embeddings[movie_idx].unsqueeze(0)
    similarities = cosine_similarity(movie_embedding, embeddings)[0]

    similar_indices = similarities.argsort()[-k-1:-1][::-1]
    recommendations = [index_to_title[idx] for idx in similar_indices if idx in index_to_title]
    return recommendations

def test_recommendation():
    embeddings, node_mapping = load_saved_data()
    if embeddings is None or node_mapping is None:
        return

    test_movie = "Black Panther"
    print(f"Recommendations for '{test_movie}':")
    recommendations = recommend_movies(test_movie, embeddings, node_mapping, k=5)
    if recommendations:
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("No recommendations found.")

if __name__ == "__main__":
    try:
        data = load_data()
        print("Data loaded successfully.")
        train_gnn(data)
        test_recommendation()
    except Exception as e:
        print("Error:", e)
