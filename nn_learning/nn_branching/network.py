#! /usr/bin/env python
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from plnn.proxlp_solver.utils import LinearOp, ConvOp, BatchConvOp, BatchLinearOp
from tools.custom_torch_modules import Flatten

'''
Training test file for the deep model transferrability studies
1. added norm weight in the backward pass
2. has been tested on wide and deep networks --- satisfactory performances

Currently treated as the correct model 
'''

class EmbedLayerUpdate(nn.Module):
    '''
    this class updates embeded layer one time
    '''
    def __init__(self, p):
        super(EmbedLayerUpdate, self).__init__()
        self.p = p

        #inputs
        # self.inp_f = nn.Linear(3,p)
        # self.inp_f_1 = nn.Linear(p,p)

        # for activation nodes
        self.fc1 = nn.Linear(5, p)
        self.fc1_1 = nn.Linear(p, p)
        self.fc2 = nn.Linear(2*p, p)     
        self.fc2_1 = nn.Linear(p, p)  


    def forward(self, lower_bounds, upper_bounds, mask, primal_score, secondary_score, layers, mu):

        # NOTE: All bounds should be at the same size as the layer outputs
        #       We have assumed that the last property layer is linear       

        ## FORWARD PASS
        batch_size = len(lower_bounds[0])
        p = self.p
        relu_idx = 0
        bound_idx = 1
        out_features = [-1]+ th.tensor(lower_bounds[0][0].size()).tolist()

        for layer_idx, layer in enumerate(layers[:-1]):
            if type(layer) in [BatchConvOp, ConvOp, nn.Conv2d]:
                layer_weight = layer.weight if type(layer) in [nn.Conv2d] else layer.weights
                if type(layer) is BatchConvOp:
                    layer_bias = layer.unconditioned_bias.detach().view(-1)
                elif type(layer) is ConvOp:
                    layer_bias = layer.bias.view(-1)
                else:
                    layer_bias = layer.bias

                #reshape 
                mu_inp = th.cat([i for i in mu[relu_idx]], 1)
                mu_inp = th.t(mu_inp).reshape(out_features)
                nb_embeddings_pre = F.conv2d(mu_inp, layer_weight, bias=layer_bias,
                                        stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
                # record and transfer back
                out_features = th.tensor(nb_embeddings_pre.size()).tolist()
                nb_embeddings_pre = nb_embeddings_pre.reshape(out_features[0],-1)
                nb_embeddings_pre = th.cat([nb_embeddings_pre[i*p:(1+i)*p] for i in range(batch_size)], 1)
                nb_embeddings_pre = th.t(nb_embeddings_pre)
                layer_lower_pre = lower_bounds[bound_idx].view(-1)
                layer_upper_pre = upper_bounds[bound_idx].view(-1)
                bound_idx += 1

            elif type(layer) in [nn.Linear, LinearOp, BatchLinearOp]: 
                layer_weight = layer.weight if type(layer) in [nn.Linear] else layer.weights
                layer_bias = layer.bias.unsqueeze(-1).repeat(1, p)
                nb_embeddings_pre = layer_weight @ mu[relu_idx] + layer_bias
                nb_embeddings_pre = th.cat([i for i in nb_embeddings_pre], 0)
                
                out_features = [-1, nb_embeddings_pre.size()[0]]
                layer_lower_pre = lower_bounds[bound_idx].view(-1)
                layer_upper_pre = upper_bounds[bound_idx].view(-1)
                bound_idx += 1

            elif type(layer) is nn.ReLU:
            # node features
                upper_ratio, lower_ratio, diff = compute_ratio(layer_lower_pre, layer_upper_pre)
                #import pdb; pdb.set_trace()

                # collecting information
                primal_score_inp = th.cat([i for i in primal_score[relu_idx]], 0)
                secondary_score_inp = th.cat([i for i in secondary_score[relu_idx]], 0)
                nb_informations = th.cat([upper_ratio.unsqueeze(-1), 
                                            lower_ratio.unsqueeze(-1),
                                            diff.unsqueeze(-1), 
                                            primal_score_inp.unsqueeze(-1),
                                            secondary_score_inp.unsqueeze(-1)], 1)
                nb_embeddings = self.fc1_1(F.relu(self.fc1(nb_informations)))
      
                # embedding updates
                embedding_informations = th.cat([nb_embeddings_pre, nb_embeddings], 1)
                nb_embeddings = self.fc2_1(F.relu(self.fc2(embedding_informations)))
                
                # attention
                attention_map = th.cat([i for i in mask[relu_idx]], 0).reshape(-1, 1)
                nb_embeddings = nb_embeddings * attention_map

                relu_idx += 1
                mu[relu_idx] = nb_embeddings.reshape(mu[relu_idx].size())

            elif type(layer) in [nn.Flatten, Flatten]:
                out_features = [-1] + th.tensor(lower_bounds[bound_idx].size()).tolist() 
                pass
            else:
                raise NotImplementedError
            
        return mu





class EmbedUpdates(nn.Module):
    '''
    this class updates embeding vectors from t=1 and t=T
    '''

    def __init__(self, p):
        '''
        p_list contains the input and output dimensions of embedding vectors for all layers
        len(p_list) = T+1
        '''
        super(EmbedUpdates, self).__init__()
        self.p = p
        self.update = EmbedLayerUpdate(p)


    def forward(self, lower_bounds, upper_bounds, mask, layers, primal_score, secondary_score):
        mu = init_mu(lower_bounds, self.p)
        mu = self.update(lower_bounds, upper_bounds, mask, primal_score, secondary_score, layers, mu)
        return mu



class ComputeFinalScore(nn.Module):
    '''
    this class computes a final score for each node

    p: the dimension of embedding vectors at the final stage
    '''
    def __init__(self, p):
        super(ComputeFinalScore,self).__init__()
        self.p = p
        self.fnode = nn.Linear(p, p)
        self.fscore = nn.Linear(p, 2)


    def forward(self, mu, primal_score, secondary_score, mask):
        nn_scores = []
        for i, batch_layer_mu in enumerate(mu[1:-1]):
            layer_mu = th.cat([i for i in batch_layer_mu], 0)
            batch_mask = th.cat([m for m in mask[i]], 0).reshape(-1, 1)
            scores = self.fscore(F.relu(self.fnode(layer_mu)))
            if not self.training:
                batch_primal = th.cat([s for s in primal_score[i]], 0).reshape(-1, 1)
                batch_second = th.cat([s for s in secondary_score[i]], 0).reshape(-1, 1)
                base_scores = th.cat([batch_primal, batch_second], -1)
                scores = base_scores * (1 + scores) * batch_mask
            scores = scores.reshape(batch_layer_mu.shape[0], batch_layer_mu.shape[1], -1)
            nn_scores.append(scores)

        return nn_scores
        

class ExpNet(nn.Module):
    def __init__(self, p):
        super(ExpNet, self).__init__()
        self.EmbedUpdates = EmbedUpdates(p)
        self.ComputeFinalScore = ComputeFinalScore(p)

    def forward(self, lower_bounds, upper_bounds, mask, layers, primal_score, secondary_score):
        mu = self.EmbedUpdates(lower_bounds, upper_bounds, mask, layers, primal_score, secondary_score)
        scores = self.ComputeFinalScore(mu, primal_score, secondary_score, mask)

        return scores


def init_mu(lower_bounds_all, p):
    mu = []
    batch_size = len(lower_bounds_all[0])
    for i in lower_bounds_all:
        required_size = i[0].view(-1).size()
        mus_current_layer = lower_bounds_all[0].new_full((batch_size,required_size[0],p), fill_value=1.0) 
        mu.append(mus_current_layer)
    
    return mu


def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound - F.relu(lower_bound)
    upper_temp = F.relu(upper_bound)
    diff = upper_temp-lower_temp
    upper_ratio = upper_temp / (diff+1e-8)
    lower_ratio = 1 - upper_ratio

    return upper_ratio, lower_ratio, diff


