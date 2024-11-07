import sys
import utils
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import os
import pdb
import pandas as pd
import torch
import pdb
import nopdb

def train_step(model, criterion, optimizer, data_loader, args):

    model.train()
    step = 0
    loss =.0
    all_length =0
    
    for hadm_id, ts, diag, entity,label, _ in data_loader :
        
        all_length += len(label)
        optimizer.zero_grad()

        ts = ts.to(args.device)
        diag = diag.to(args.device)
        if len(entity) >=1:
            entity = entity.to(args.device)
        label = label.to(args.device)

        logits , aux_logits,_,_ = model(ts,entity)
        logits = torch.sigmoid(logits)

        
            

        _loss = criterion(logits,label)
        
        aux_logits = torch.sigmoid(aux_logits)
        aux_los = criterion(aux_logits,diag)
        _loss= 0.5 *_loss +  0.5 * aux_los

        _loss.backward()
        loss += _loss.item()
        optimizer.step()

        step += 1 

        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
        print("Process Training Batch: [{}/{}]".format(step,
                                                       len(data_loader)))

    return loss  / all_length

def eval_step(model, criterion, data_loader, args):

    model.eval()
    step = 0
    loss =.0
    all_length =0

    pred_list=[]
    true_list=[]

    for hadm_id, ts, diag, entity,label,_ in data_loader :
        
        all_length += len(label)
        
        ts = ts.to(args.device)
        diag = diag.to(args.device)
        if len(entity) >=1:
            entity = entity.to(args.device)
        label = label.to(args.device)#.unsqueeze(1)
        
        logits, aux_logits,_,_ = model(ts,entity)
        logits = torch.sigmoid(logits)
      


        _loss = criterion(logits,label)

        loss += _loss.item()

        pred_list.extend(logits.cpu().detach().numpy())
        true_list.extend(label.cpu().numpy())
        

        step += 1 

        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
        print("Process valid Batch: [{}/{}]".format(step,
                                                       len(data_loader)))


    return loss / all_length , np.array(pred_list).squeeze(), np.array(true_list).squeeze()


def predict_step(model, data_loader, args):
    model.eval()
    step = 0
    loss =.0
    all_length =0

    all_pred =[]

    for hadm_id, ts, ccs, entity,label,  raw_sample in data_loader:

        all_length += len(label)
        
        ts = ts.to(args.device)
        ccs = ccs.to(args.device)
        label = label.to(args.device)#.unsqueeze(1)
        entity = entity.to(args.device)
        
        logits,aux_logit,entity_feature, fusion_feature  = model(ts,ccs,entity )
    
        predicts = torch.sigmoid(logits).cpu().detach().numpy()
        label = label.cpu().numpy()
        #aux_logit = torch.sigmoid(aux_logit).cpu().detach().numpy()
        fusion_feature = fusion_feature.cpu().detach().numpy()
        entity_feature = entity_feature.cpu().detach().numpy()
        ccs = ccs.cpu().numpy()
        entity= entity.cpu()
        hadm_id = hadm_id.numpy()
        for i in range(len(label)):
            
            all_pred.append([hadm_id[i],label[i], predicts[i] ,ccs[i]])
            
    return all_pred