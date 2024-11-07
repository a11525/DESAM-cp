import pandas as pd
import os
import numpy as np
import tqdm
import warnings
warnings.filterwarnings('ignore')
import ast
from transformers import AutoTokenizer, AutoModel
import torch
from multiprocessing import Pool
import pdb


if __name__ == "__main__":

    pretrained_path ='emilyalsentzer/Bio_ClinicalBERT'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)  
    model = AutoModel.from_pretrained(pretrained_path).cuda() #.cuda(1)
    data_path = '../data/'
    text_pd= pd.read_pickle(data_path+'for_bert_text.pkl')
    all_text=text_pd['preprocessed_text'].values
    all_embs=[]
    
    with torch.no_grad():

        for i in tqdm.tqdm(np.arange(0, len(all_text)), total=len(all_text)):
            toks = tokenizer.batch_encode_plus(all_text[i], 
                                            padding="max_length", 
                                            max_length=256, 
                                            truncation=True,
                                            return_tensors="pt")
            toks_cuda = {}
            for k,v in toks.items():
                toks_cuda[k] = v.cuda()
            cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
            all_embs.append(cls_rep.cpu().detach().numpy())
    #pdb.set_trace()
    all_embs = np.concatenate(all_embs, axis=0)

    save_path=data_path
    np.save(save_path+'sentence_to_cbert_cls.npy',all_embs)