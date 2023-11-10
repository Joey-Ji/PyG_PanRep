# PanRep API Documentation

## Overview

**PanRep** is a Graph Neural Network (GNN) model designed for the unsupervised learning of universal node representations in heterogeneous graphs. It achieves state-of-the-art performance in node classification and link prediction tasks and can be fine-tuned with limited labels.

## models.PanRep

```python
PanRep(
    edge_index_dict: Dict[EdgeType, Tensor],
    embedding_dim: int,
    num_clusters: int,
    node_features: Tensor
)
```

Parameters:
edge_index_dict: A dictionary with keys as EdgeType and values as Tensor. Each key-value pair represents the connectivity of a different type of edge in the heterogeneous graph.

embedding_dim: The size of each embedding vector.

num_clusters: The number of clusters to form.

node_features: A tensor containing features of the nodes (shape: N-by-(num_features)).

```python
reset_parameters(self)
```
Resets all learnable parameters within the cr_head linear layer. It should be called before training begins.

```python
forward(self) -> Tensor
```
Passes the node embeddings through a linear layer followed by a sigmoid activation function to produce the predicted clusters.


```python
loss(self, predicted_clusters: Tensor) -> Tensor
```
Calculates the loss between the predicted clusters and true clusters obtained from KMeans clustering on the node features.