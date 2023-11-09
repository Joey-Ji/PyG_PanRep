from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.sparse import index2ptr

from kmeans_pytorch import kmeans

class PanRep(torch.nn.Module):
    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, Tensor],
        embedding_dim: int,
        num_clusters: int,
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
        count = 0
        for key in num_nodes_dict:
            count += num_nodes_dict[key] 

        self.embedding_dim = embedding_dim
        self.node_embeddings = Embedding(num_embeddings=count, embedding_dim=embedding_dim)
        self.num_clusters = num_clusters
        self.cr_matrix = torch.nn.Parameter(torch.randn(embedding_dim, num_clusters))

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.node_embeddings.reset_parameters()
        self.cr_matrix.reset_parameters()

    def forward():

    def loss(self) -> Tensor{

    }