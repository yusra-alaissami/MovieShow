import torch
from train_gnn import GNN
from data_loader import load_data
import torch.nn.functional as F

# Charger les données
try:
    print("Chargement des données...")
    data = load_data()
    print("Données chargées avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement des données : {e}")
    exit()

# Vérifier les dimensions des données
print(f"Features des nœuds : {data.x.shape}")
print(f"Edge index : {data.edge_index.shape}")

# Vérifier la cohérence des indices dans edge_index
if data.edge_index.max() >= data.x.size(0):
    raise ValueError("edge_index contient des indices de nœuds hors limites.")

# Ajouter des labels fictifs pour les nœuds (nécessaire pour l'entraînement)
if not hasattr(data, 'y') or data.y is None:
    print("Ajout de labels fictifs pour les nœuds...")
    data.y = torch.randint(0, 2, (data.x.size(0),))  # Exemple : 2 classes (0 ou 1)

# Diviser les données en train/test
train_mask = torch.rand(data.x.size(0)) < 0.8  # 80% pour l'entraînement
test_mask = ~train_mask  # 20% pour le test

# Initialiser le modèle
input_dim = data.x.size(1)  # Nombre de features par nœud
hidden_dim = 16  # Taille des couches cachées
output_dim = 2   # Nombre de classes
model = GNN(input_dim, hidden_dim, output_dim)

# Optimiseur
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Fonction d'évaluation
def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
        return acc

# Entraîner le modèle
print("Entraînement du modèle...")
for epoch in range(20):  # Réduire les epochs pour les tests rapides
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    # Évaluer les performances
    train_acc = evaluate(model, data, train_mask)
    test_acc = evaluate(model, data, test_mask)
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# Sauvegarder le modèle
torch.save({
    'model_state_dict': model.state_dict(),
    'node_mapping': data.node_mapping
}, "gnn_model_test.pth")
print("Modèle sauvegardé sous 'gnn_model_test.pth'.")
