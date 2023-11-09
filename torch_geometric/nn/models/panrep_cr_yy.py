from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.sparse import index2ptr

from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans

import tqdm

class PanRep(torch.nn.Module):
    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, Tensor],
        embedding_dim: int,
        num_clusters: int,
        node_features: Tensor,  # N-by-(num_features)
    ):
        super().__init__()

        # TODO:: Have to count the number of nodes
        if num_nodes_dict is None:
            num_nodes_dict = {}
            for keys, edge_index in edge_index_dict.items():
                key = keys[0]
                N = int(edge_index[0].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

                key = keys[-1]
                N = int(edge_index[1].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))
        total_num_nodes = 0
        for key in num_nodes_dict:
            total_num_nodes += num_nodes_dict[key] 

        self.embedding_dim = embedding_dim
        self.node_embeddings = nn.Parameter(torch.randn(total_num_nodes, embedding_dim), requires_grad=True)
        self.num_clusters = num_clusters
        #self.cr_matrix = torch.nn.Parameter(torch.randn(embedding_dim, num_clusters))
        self.cr_head = nn.Linear(embedding_dim, num_clusters)
        self.node_features = node_features

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.cr_head.reset_parameters()

    def forward(self):
        return F.sigmoid(self.cr_head(self.node_embeddings.t()))

    def loss(self, predicted_clusters): #predicted_clusters should be the output of the forward method
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(self.node_features.detach().cpu().numpy())
        # Convert the cluster labels to a one-hot encoding
        y_pred = kmeans.labels_
        y_one_hot = torch.eye(self.num_clusters)[y_pred]
        true_clusters = y_one_hot.to(self.node_features.device)

        return -torch.sum(true_clusters * torch.log(predicted_clusters + 1e-15)) 
