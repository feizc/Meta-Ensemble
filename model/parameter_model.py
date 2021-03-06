import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast
from torch.nn import functional as F 
import numpy as np 



class FeedForward(nn.Module): 
    def __init__(self, d_model, d_ff, dropout=.1):
        super(FeedForward, self).__init__() 
        self.fc1 = nn.Linear(d_model, d_ff) 
        self.fc2 = nn.Linear(d_ff, d_model) 
        self.dropout = nn.Dropout(p=dropout) 
        self.dropout_2 = nn.Dropout(p=dropout) 
        self.layer_norm = nn.LayerNorm(d_model) 
    
    def forward(self, input): 
        out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
        out = self.dropout(out)
        out = self.layer_norm(input + out) 
        return out 



# module name can not contain "."
def module_name_refine(module_name): 
    module_name = module_name.split('.')
    refine_module_name = ''
    for seg in module_name: 
        refine_module_name += seg 
    return refine_module_name



# linear layer for parameter prediction 
class ParameterProject(nn.Module): 
    # project base learner to paramer distillation 
    def __init__(self, weight_size_dict):
        super(ParameterProject, self).__init__() 
        self.models = nn.ModuleDict()

        for key in weight_size_dict: 
            self.models[module_name_refine(key)] = nn.Linear(weight_size_dict[key][1], weight_size_dict[key][1]//2)
    
    def forward(self, combine_weight_dict): 
        for key in combine_weight_dict.keys(): 
            #print(self.models[module_name_refine(key)])
            #print(key, combine_weight_dict[key].size())
            combine_weight_dict[key] = self.models[module_name_refine(key)](combine_weight_dict[key]) 
        return combine_weight_dict



# lib for transformer 
class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h=8, seq, seq,) 
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out




class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out




class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out




class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff




# construct the embedding from position, and segment  
class TransformerEmbeddings(nn.Module): 
    def __init__(self, max_pos_len=512, type_vocab_size=2, d_model=512):
        super().__init__() 
        self.position_embeddings = nn.Embedding(max_pos_len, d_model) 
        self.segment_type_embedding = nn.Embedding(type_vocab_size, d_model) 
        self.register_buffer("position_ids", torch.arange(max_pos_len).expand((1, -1)))

    def forward(self, input_embedding): 
        input_shape = input_embedding.size() # (bsz, n_o, d_model)
        seq_len = input_shape[1] 

        segment_ids = torch.gt(torch.arange(seq_len), seq_len//2).long().unsqueeze(0)
        segment_type_embeddings = self.segment_type_embedding(segment_ids) 

        position_ids = self.position_ids[:, :seq_len] 
        position_embeddings = self.position_embeddings(position_ids)  
        return input_embedding + segment_type_embeddings + position_embeddings 




# Original transformer for ensemble parameter prediction 
class Transformer(nn.Module): 
    def __init__(self, N, padding_idx, size_dict, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(Transformer, self).__init__() 
        # self.fc = nn.Linear(3*3*3, d_model) 
        self.fc_dict = nn.ModuleDict() 
        self.inverse_fc_dict = nn.ModuleDict() 

        for key in size_dict: 
            self.fc_dict[module_name_refine(key)] = nn.Linear(size_dict[key][1], d_model) 
            self.inverse_fc_dict[module_name_refine(key)] = nn.Linear(d_model, size_dict[key][1])
    
        self.dropout = nn.Dropout(p=dropout) 
        self.layer_norm = nn.LayerNorm(d_model) 

        self.transforms = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                        for _ in range(N)]) 
        
        self.embeddings = TransformerEmbeddings() 
        self.padding_idx = padding_idx 

    def forward(self, input, named_layer): 
        out = self.fc_dict[module_name_refine(named_layer)](input) 
        out = F.relu(out) 
        out = self.embeddings(out)
        out = self.dropout(out) 
        out = self.layer_norm(out)  

        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(out, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        # outs = [] 
        for l in self.transforms:
            out = l(out, out, out, attention_mask)
            # outs.append(out.unsqueeze(1))
        seq_len = out.size()[1]//2 
        out = out[:, :seq_len, :]  # (bsz, n_o, d_in)
        # outs = torch.cat(outs, 1) # (b_s, encoder layer, seq_len, d_in) 
        
        out = self.inverse_fc_dict[module_name_refine(named_layer)](out) # (bsz, n_o, k*k*n_i) 
        return out


