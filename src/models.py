import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Any, Union, Callable

class EntityAttentionModel(nn.Module):
    def __init__(self, args):
        super(EntityAttentionModel, self).__init__()

        self.args =args
        
        self.TS_encoder = nn.GRU(args.TS_dimension_size, 128,batch_first=True, dropout= args.dropout_rate,bidirectional=True)
       
        self.Entity_encoder = EntitySelfAttention(args)
        self.aux_fc1 = nn.Linear(768,256)
        self.aux_fc2 = nn.Linear(256,251)

        self.fc1 = nn.Linear(768 +256 ,256)
        self.last_layer = nn.Linear(256,args.label_dim)
        

        self.dropout = nn.Dropout(p=args.dropout_rate)
        if args.act_func == 'relu' :
            self.act_func = nn.ReLU()
        elif args.act_func == 'tanh':
            self.act_func = nn.Tanh()

        if args.act_init:
            if args.act_func=='relu':
                init_relu(self.fc1)
                init_relu(self.last_layer)
            elif args.act_func=='tanh':
                init_tanh(self.fc1)
                init_tanh(self.last_layer)
    def forward(self,  ts, entity):
        
        ts_seq = self.TS_encoder(ts) 
        
        ts_hiddens = ts_seq[0][:, -1, :] # batch, 104


        
        if self.args.use_entity:
            entity_feature = self.Entity_encoder(entity)
        else:
            entity_feature = entity.mean(1)
            entity_feature = self.act_func(self.encoder_for_bert(entity_feature))

        fusion_feature = torch.concat([ts_hiddens, entity_feature], axis=1)
        
        
        x = self.dropout(self.act_func(self.fc1(fusion_feature)))
        x = self.last_layer(x)

        aux_feature=self.dropout(self.act_func(self.aux_fc1(entity_feature)))
        #aux_feature = self.dropout(self.act_func(self.aux_fc1(fusion_feature)))
        aux_feature = self.aux_fc2(aux_feature)
        
        if self.args.task != 'predict':
            entity_feature =0
            fusion_feature =0

        return x, aux_feature , entity_feature, fusion_feature


class EntitySelfAttention(nn.Module):
    def __init__(self,args):
        super(EntitySelfAttention,self).__init__()
        self.args=args
        self.input_dim = 768
        self.last_layer_bool = args.last_layer

        code_encoder_layer = MyTransformerEncoderLayer(768, args.nhead, args.hidden_dim, args.dropout_rate, batch_first=True)
        self.code_encoder = nn.TransformerEncoder(code_encoder_layer, num_layers= args.num_layer)

    def forward(self, x ):
        
        pad_mask = ((x==0).float().sum(-1) !=0).float()
        x = self.code_encoder(x,src_key_padding_mask=pad_mask) 
    
        mask = (1-pad_mask.unsqueeze(-1))
        x = x * mask
        x = x.sum(1)/ mask.sum(1)

                
        return x  


class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(
            query=x, key=x, value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )[0]
        return self.dropout1(x)

def init_hidden_he(layer):
    layer.apply(init_relu)
def init_hidden_ta(layer):
    layer.apply(init_tanh)

def init_relu(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, 2 ** 0.5)

def init_tanh(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, 5 / 3)

def create_padding_mask(x):
    mask = (x == 0).float()
    return mask.unsqueeze(1).unsqueeze(2)