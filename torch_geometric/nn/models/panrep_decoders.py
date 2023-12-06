import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.cluster import KMeans


class ClusterRecoverDecoderHomo(nn.Module):
    def __init__(self, n_cluster, in_size, h_dim,
 activation=nn.ReLU(), single_layer=True):
        '''

        :param out_size_dict:
        :param in_size:
        :param h_dim:
        :param masked_node_types: Node types to not penalize for reconstruction
        :param activation:
        '''

        super(ClusterRecoverDecoderHomo, self).__init__()
        # W_r for each node
        self.activation=activation
        self.h_dim=h_dim
        self.n_cluster=n_cluster

        self.single_layer=single_layer
        layers=[]
        if self.single_layer:
                layers.append(nn.Linear(in_size, n_cluster, bias=False))
        else:
                layers.append(nn.Linear( in_size, self.h_dim))
                layers.append(activation)
                layers.append(nn.Linear(self.h_dim, n_cluster, bias=False))
        self.weight=nn.Sequential(*layers)

    def loss_function(self,predicted_clusters,true_clusters):
        # cross entropy
        return -torch.sum(true_clusters * torch.log(predicted_clusters + 1e-15)) 


    def forward(self, g, node_embed):
        reconstructed=self.weight(node_embed)
        # compute k mean clustering 
        kmeans = KMeans(n_clusters=self.n_cluster)
        kmeans.fit(g.node_features.detach().cpu().numpy())
        # Convert the cluster labels to a one-hot encoding
        y_pred = kmeans.labels_
        y_one_hot = torch.eye(self.n_cluster)[y_pred]
        true_clusters = y_one_hot.to(g.node_features.device)
        # got matrix C

        loss = self.loss_function(reconstructed, true_clusters)
        return loss