import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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
        print("\nEmbeddings for all nodes:")
        for i, embedding in enumerate(embeddings):
            print(f"Node {i}: {embedding.tolist()}")

    torch.save({'embeddings': embeddings, 'node_mapping': data.node_mapping}, "node_embeddings.pth")
    print("\nEmbeddings saved successfully.")

