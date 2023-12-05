'''
This file contains the PanRep model.

'''

import torch.nn as nn
import torch as th
import dgl.function as fn
from panrep_encoder import EncoderHGT
from panrep_decoders import NodeMotifDecoder,MultipleAttributeDecoder\
    ,MutualInformationDiscriminator
import dgl



class PanRepHetero(nn.Module):
    def __init__(self,
                 encoder,
                 decoders):
        super(PanRepHetero, self).__init__()
        self.decoders=decoders
        self.encoder=encoder
        
    def forward(self, g, masked_nodes):
        positive = self.encoder(g,corrupt=False)
        loss=0

        for decoderName, decoderModel in self.decoders.items:

            if decoderName=='mid':
                negative = self.encoder(g,corrupt=True)
                infomax_loss = decoderModel(positive, negative)
                loss += infomax_loss

            if decoderName=='crd':
                reconstruct_loss = decoderModel(g,positive,masked_nodes=masked_nodes)
                loss += reconstruct_loss

            if decoderName == 'nmd':
                motif_loss = decoderModel(g,positive)
                loss += motif_loss

        return loss, positive