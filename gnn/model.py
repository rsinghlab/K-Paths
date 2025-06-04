import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN(torch.nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN) with learned node embeddings.

    The model uses an initial embedding layer, a projection to 768 dimensions,
    followed by multiple RGCNConv layers with batch normalization and dropout.

    Args:
        num_nodes (int): Total number of nodes in the graph.
        embedding_dim (int): Size of the initial node embeddings.
        hidden_dim (int): Hidden dimension of each RGCNConv layer.
        num_relations (int): Number of edge types (relations) in the graph.
        num_layers (int): Number of RGCNConv layers (default: 3).
        num_bases (int): Number of basis vectors to use in RGCN (default: 2).
        dropout (float): Dropout rate applied after each layer (default: 0.5).
    """
    def __init__(
        self,
        num_nodes,
        embedding_dim,
        hidden_dim,
        num_relations,
        num_layers=3,
        num_bases=2,
        dropout=0.5,
    ):
        super(RGCN, self).__init__()
        self.embeddings = torch.nn.Embedding(num_nodes, embedding_dim)
        self.projection = torch.nn.Linear(embedding_dim, 768)  # Project to 768 dimensions
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                conv = RGCNConv(768, hidden_dim, num_relations, num_bases=num_bases)
            else:
                conv = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
            self.convs.append(conv)
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, edge_index, edge_type):
        """
        Forward pass of the RGCN.

        Args:
            edge_index (Tensor): Edge indices of shape [2, num_edges].
            edge_type (Tensor): Edge type IDs of shape [num_edges].

        Returns:
            Tensor: Final node embeddings of shape [num_nodes, hidden_dim].
        """
        x = self.embeddings.weight
        x = F.normalize(x, p=2, dim=-1)
        x = self.projection(x)  # Project embeddings to 768 dimensions

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_type)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return x

class EdgeClassifier(torch.nn.Module):
    """
    Edge classifier that predicts the relation type between node pairs.

    The classifier takes concatenated node embeddings of edge endpoints and
    passes them through a two-layer MLP.

    Args:
        node_embedding_dim (int): Dimensionality of input node embeddings.
        hidden_dim (int): Hidden size of the MLP.
        num_classes (int): Number of edge (relation) classes to predict.
    """
    def __init__(self, node_embedding_dim, hidden_dim, num_classes):
        super(EdgeClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(2 * node_embedding_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, node_embeddings, edge_index):
        """
        Predicts edge labels given node embeddings and edge indices.

        Args:
            node_embeddings (Tensor): Node embeddings of shape [num_nodes, embedding_dim].
            edge_index (Tensor): Edge indices of shape [2, num_edges].

        Returns:
            Tensor: Predicted logits of shape [num_edges, num_classes].
        """
        h = node_embeddings[edge_index[0]]  # [num_edges, embedding_dim]
        t = node_embeddings[edge_index[1]]  # [num_edges, embedding_dim]
        edge_emb = torch.cat([h, t], dim=1)  # [num_edges, 2 * embedding_dim]
        out = self.fc1(edge_emb)
        out = self.relu(out)
        out = self.fc2(out)
        return out
