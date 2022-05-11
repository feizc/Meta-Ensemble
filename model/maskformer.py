import torch 
import torch.nn as nn 
from .parameter_model import EncoderLayer, module_name_refine, TransformerEmbeddings 
from torch.nn import functional as F 



class MaskformerConfig(): 
    # parameter settings for Maskformer 
    def __init__(
        self, 
        weight_dict, 
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="gelu",
        hidden_dropout_prob=0.1, 
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        type_vocab_size=3,  # [cross], model 1, model 2
        max_position_embeddings=2048,
        pad_token_id=0,
        mask_prob=0.7, 
        mask_soft2hard=True, 
        shift_flag=False, 
        shift_step=1, # shift step for consistency loss
    ):
        self.weight_dict = weight_dict
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps 
        self.type_vocab_size = type_vocab_size 
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id 
        self.mask_prob = mask_prob 
        self.mask_soft2hard = mask_soft2hard 
        self.shift_flag = shift_flag 
        self.shift_step = shift_step 





class Maskformer(nn.Module): 
    def __init__(self, config):
        super(Maskformer, self).__init__() 
        self.config = config 
        self.max_position_embeddings = config.max_position_embeddings 
        self.mask_prob = config.mask_prob 
        self.weight_dict = config.weight_dict 
        self.padding_idx = 0

        self.fc_dict = nn.ModuleDict() 
        self.inverse_fc_dict = nn.ModuleDict() 

        for key in self.weight_dict: 
            self.fc_dict[module_name_refine(key)] = nn.Linear(self.weight_dict[key][1], config.hidden_size) 
            self.inverse_fc_dict[module_name_refine(key)] = nn.Linear(config.hidden_size, self.weight_dict[key][1])

        self.embeddings = TransformerEmbeddings(max_pos_len=self.max_position_embeddings) 
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob) 
        self.layer_norm = nn.LayerNorm(config.hidden_size) 

        self.d_k = config.hidden_size // config.num_attention_heads 
        self.d_v = self.d_k 
        self.transforms = nn.ModuleList([EncoderLayer(config.hidden_size, self.d_k, self.d_v, config.num_attention_heads, 
                                                  config.intermediate_size, config.attention_probs_dropout_prob,
                                                  identity_map_reordering=False,)
                                        for _ in range(config.num_hidden_layers)]) 
        
        self.mask_soft2hard = config.mask_soft2hard 
        self.learn_mask_attention = nn.Embedding(self.max_position_embeddings*self.max_position_embeddings, 1) 
        self.sigmoid = nn.Sigmoid() 
        self.mse_loss = nn.MSELoss()
    
    def forward(self, input_weight, named_layer):  
        out = self.fc_dict[module_name_refine(named_layer)](input_weight) 
        out = F.relu(out) 
        out = self.embeddings(out)
        out = self.dropout(out) 
        out = self.layer_norm(out)  

        device = input_weight.device 
        input_weight = self.fc_dict[module_name_refine(named_layer)](input_weight) 
        bsz, seq_len = input_weight.size()[:-1]  # (bsz, seq_len)
        
        mask_attention_len = self.max_position_embeddings 
        learn_mask = self.learn_mask_attention.weight.reshape(mask_attention_len, mask_attention_len) 
        learn_mask = self.sigmoid(learn_mask) 
        # current weight token can see itself 
        diag_mask = torch.diag(torch.ones(mask_attention_len)).to(device) 
        weight_attention = (1. - diag_mask) * learn_mask 
        learn_mask = diag_mask + weight_attention 

        if self.config.mask_soft2hard: 
            learn_mask = (learn_mask >= 0.5)*1.0
            learn_mask = learn_mask.to(device) 
            learn_mask.requires_grad = False 
        
        cut_learn_mask = learn_mask[:seq_len, :seq_len].unsqueeze(0).repeat(bsz, 1, 1) # (bsz, 1, seq, seq)
        attention_mask = cut_learn_mask.unsqueeze(1)
        
        if self.config.shift_flag == True: 
            combine_learn_mask = torch.cat([learn_mask, learn_mask], dim=0) 
            shift_cut_learn_mask = combine_learn_mask[self.config.shift_step: seq_len+self.config.shift_step, :seq_len].unsqueeze(0).repeat(bsz, 1, 1) 
            shift_attention_mask = shift_cut_learn_mask.unsqueeze(1) 

            shift_out = out 
            for l in self.transforms:
                shift_out = l(shift_out, shift_out, shift_out, shift_attention_mask)
            out_len = shift_out.size()[1]//2 
            shift_out = shift_out[:, :out_len, :] 
            shift_out = self.inverse_fc_dict[module_name_refine(named_layer)](shift_out)

        for l in self.transforms:
            out = l(out, out, out, attention_mask)
            # outs.append(out.unsqueeze(1))
        out_len = out.size()[1]//2 
        out = out[:, :out_len, :]  # (bsz, n_o, d_in)
        # outs = torch.cat(outs, 1) # (b_s, encoder layer, seq_len, d_in) 
        
        out = self.inverse_fc_dict[module_name_refine(named_layer)](out) # (bsz, n_o, k*k*n_i) 

        if self.config.shift_flag == True: 
            return (out, shift_out)
        else:
            return out  


    def get_mask_loss(self, attention_mask): 
        mask_loss = 0 
        mask_loss += (torch.mean(torch.abs(attention_mask))) 
        return mask_loss 


    def get_shift_consistency_loss(self, out, shift_out): 
        shift_loss = 0 
        shift_loss += self.mse_loss(out, shift_out) 
        return shift_loss 


    def freeze_backbone(self, freeze=True): 
        for _, p in self.transforms.named_parameters(): 
            p.requires_grad = not freeze 






