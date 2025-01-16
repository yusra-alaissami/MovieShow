# Importations nécessaires
from neo4j import GraphDatabase  # Pour se connecter et interagir avec Neo4j
import torch  # Pour la gestion des tenseurs et des calculs
from torch_geometric.data import Data  # Pour manipuler les données de graphes
from sklearn.preprocessing import LabelEncoder  # Pour encoder les caractéristiques textuelles

def load_data():
    # Configuration Neo4j
    uri = "bolt://127.0.0.1:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "Nemejujepas"))

    with driver.session() as session:
        # Charger les films
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

        # Charger les relations (uniquement entre les films)
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

        # Conversion des relations en tenseurs
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long) if edge_index[0] else torch.empty((2, 0), dtype=torch.long)

        # Encodage des caractéristiques des nœuds
        le = LabelEncoder()
        if node_features:
            node_features = le.fit_transform(node_features)
            node_features_tensor = torch.tensor(node_features, dtype=torch.float).unsqueeze(1)
        else:
            node_features_tensor = torch.empty((0, 1), dtype=torch.float)

        # Vérification des correspondances
        if node_features_tensor.size(0) != len(node_mapping):
            raise ValueError("Mismatch between node features and mapping.")

        # Création de l'objet Data pour PyTorch Geometric
        data = Data(x=node_features_tensor, edge_index=edge_index_tensor)
        data.node_mapping = node_mapping
        return data
