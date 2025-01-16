from neo4j import GraphDatabase
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Neo4j connection
uri = "bolt://127.0.0.1:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "Nemejujepas"))

# Data Loader
def load_data():
    with driver.session() as session:
        # Load Movies
        movie_query = """
            MATCH (m:Movie) RETURN m.uuid AS id, m.title AS title
        """
        movies = session.run(movie_query)
        node_mapping = {}
        node_features = []
        seen_titles = set()

        print("Loading Movies...")
        for idx, record in enumerate(movies):
            if record["id"] and record["title"] not in seen_titles:
                seen_titles.add(record["title"])
                node_mapping[record["title"]] = idx
                node_features.append(record["title"])
                print(f"Loaded Movie: {record['title']} (UUID: {record['id']})")
            else:
                print(f"Ignored Duplicate or Invalid Movie: {record}")

        # Load Relations (Movies only)
        relation_query = """
            MATCH (m1:Movie)-[:RELATED_TO]->(m2:Movie)
            RETURN m1.title AS source, m2.title AS target
        """
        relations = session.run(relation_query)
        edge_index = [[], []]
        print("Loading Movie Relations...")
        for record in relations:
            source = record["source"]
            target = record["target"]
            if source in node_mapping and target in node_mapping:
                edge_index[0].append(node_mapping[source])
                edge_index[1].append(node_mapping[target])
                print(f"Loaded Relation: {source} -> {target}")
            else:
                print(f"Ignored Invalid Relation: {record}")

        # Convert to PyTorch tensors
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long) if edge_index[0] else torch.empty((2, 0), dtype=torch.long)

        # Encode movie titles to numerical features
        le = LabelEncoder()
        if node_features:
            node_features = le.fit_transform(node_features)
            node_features_tensor = torch.tensor(node_features, dtype=torch.float).unsqueeze(1)
        else:
            node_features_tensor = torch.empty((0, 1), dtype=torch.float)

        if node_features_tensor.size(0) != len(node_mapping):
            raise ValueError("Mismatch between node features and mapping.")

        data = Data(x=node_features_tensor, edge_index=edge_index_tensor)
        data.node_mapping = node_mapping
        return data

# GNN Model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_node_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.relu(x)

# Training and Saving Embeddings
def train_gnn(data):
    model = GNN(input_dim=data.x.size(1), hidden_dim=16, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data.y = torch.randint(0, 2, (data.num_nodes,))
    train_mask = torch.rand(data.num_nodes) < 0.8
    test_mask = ~train_mask

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = model.get_node_embeddings(data.x, data.edge_index)

    torch.save({'embeddings': embeddings, 'node_mapping': data.node_mapping}, "node_embeddings.pth")
    print("Embeddings saved successfully.")

# Recommendation Function
def recommend_movies(movie_title, embeddings, node_mapping, k=5):
    # Reverse mapping: index -> title
    index_to_title = {idx: title for title, idx in node_mapping.items()}

    # Find movie index using the title
    movie_idx = None
    for title, idx in node_mapping.items():
        if title == movie_title:
            movie_idx = idx
            break

    if movie_idx is None:
        print(f"Movie '{movie_title}' not found.")
        return []

    # Query for additional recommendations from Neo4j
    with driver.session() as session:
        recommendation_query = """
            

        MATCH (m1:Movie {title: $title})<-[:DIRECTED]-(d:Person)-[:DIRECTED]->(m2:Movie)
OPTIONAL MATCH (u:User)-[:LIKES]->(m1), (u)-[:LIKES]->(m3:Movie)
RETURN DISTINCT m2.title AS similar_director_movie, m3.title AS liked_movie
"""
        result = session.run(recommendation_query, title=movie_title)
        similar_director_movies = set()
        liked_movies = set()
        for record in result:
            if record["similar_director_movie"]:
                similar_director_movies.add(record["similar_director_movie"])
            if record["liked_movie"]:
                liked_movies.add(record["liked_movie"])

    # Combine the results
    related_movies = similar_director_movies.union(liked_movies)

    # Calculate cosine similarity
    movie_embedding = embeddings[movie_idx].unsqueeze(0)
    similarities = cosine_similarity(movie_embedding, embeddings)[0]

    # Filter top-k recommendations
    similar_indices = similarities.argsort()[-k-1:-1][::-1]
    recommendations = [
        index_to_title[idx] for idx in similar_indices if index_to_title[idx] in related_movies
    ]
    return recommendations

# Test Recommendation
def test_recommendation():
    embeddings, node_mapping = load_saved_data()
    if embeddings is None or node_mapping is None:
        return

    test_movie = "Inception"  # Title of the movie for testing
    print(f"Recommendations for '{test_movie}':")
    recommendations = recommend_movies(test_movie, embeddings, node_mapping, k=5)
    if recommendations:
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("No recommendations found.")

# Load Saved Data
def load_saved_data(file_path="node_embeddings.pth"):
    try:
        saved_data = torch.load(file_path)
        return saved_data['embeddings'], saved_data['node_mapping']
    except FileNotFoundError:
        print("Error: Embedding file not found.")
        return None, None

# Main Execution
if __name__ == "__main__":
    try:
        data = load_data()
        print("Data loaded successfully.")
        train_gnn(data)
        test_recommendation()
    except Exception as e:
        print("Error:", e)
