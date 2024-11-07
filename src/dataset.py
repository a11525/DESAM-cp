
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import pdb
from config import get_args
import time
import numpy as np
import pdb
class EXTRACTDataset(Dataset):
    def __init__(self, args,data_type):
        self.args = args
        split_f = args.data_path + args.split_folder +'/' +data_type
        self.hadm_ids = pd.read_pickle(split_f+'_hadm_idx.pkl')
        self.ts_data = pd.read_pickle(split_f+'_ts_simple_impute.pkl').reset_index().set_index(['hadm_id','subject_id','icustay_id','hours_in'])
        

        
        self.label = pd.read_csv(args.data_path+"label_last_visit.csv").set_index('hadm_id').loc[self.hadm_ids]
        self.entity_features =pd.read_pickle(args.data_path+'cbert_emb_by_hadm.pkl')
        self.padd_entity = np.zeros((512, 768),dtype=np.float32)

        self.ccs_label = pd.read_pickle(args.data_path+'ccs_label.pkl')
        
        have_entity=list(self.entity_features.keys())
        re_sub=[]
        for idx in self.hadm_ids:
            if idx in have_entity:
                re_sub.append(idx)

        self.hadm_ids = re_sub
        

        
    
        
        self.sort_data()

        print('Load {} data # of patients  : {} \n'.format(data_type, len(self.hadm_ids)))


    def sort_data(self):
        self.hadm_ids.sort()
        self.subject_ids = self.label.loc[self.hadm_ids]['subject_id'].values
        self.ts_data = torch.tensor(self.ts_data.loc[self.hadm_ids].values.reshape(len(self.hadm_ids),24,-1)).type(torch.float32)

        self.label = torch.tensor(self.label.loc[self.hadm_ids][['mort_hosp','mort_icu','los_3','los_7']].values).type(torch.float32)
        
        if len(self.entity_features)>0:
            try:
                self.entity_features = torch.tensor(np.array(list(map(lambda x : self.entity_make_padd(self.entity_features[x]), self.hadm_ids))))
            except:
                pdb.set_trace()
        
        self.ccs_label = torch.tensor(np.array(list(map(lambda x : self.ccs_label[x], self.hadm_ids)))).type(torch.float32)

    def label_dim(self):
        return self.label.shape[-1]

    def entity_make_padd(self, entity_feature):
        feature_padd= self.padd_entity.copy()
        
        feature_padd[:len(entity_feature[:512]),:]=entity_feature[:512]
        return feature_padd

    def __getitem__(self,index):
        row_samples =  []
        len_icd9_list =[]
        temp_subj=[]
        

        if len(self.entity_features)>=1:
            entity_features=self.entity_features[index]
        else :
            entity_features = []
       
        return self.hadm_ids[index], self.ts_data[index], self.ccs_label[index],\
            entity_features, self.label[index],   [temp_subj, row_samples,len_icd9_list]

    
    def __len__(self):
        return len(self.hadm_ids)
    

def collate_fn_vary_visits(batch):
    # batch: list of tuples (samples,  label)
    # Sort the batch in decreasing order of the number of visits
    #pdb.set_trace()
    batch = sorted(batch, key=lambda x: x[5].shape[0], reverse=True)

    # Pad the samples and create a tensor
    id_list=[]
    ts_list=[]
    diag_list=[]
    entity_list=[]
    label_list=[]
    row_list=[]
    for h_id, ts, d ,e, l, row in batch:
        id_list.append(h_id)
        ts_list.append(ts)
        diag_list.append(d)
        entity_list.append(e)
        label_list.append(torch.tensor(l).reshape(1,-1))
        row_list.append(row)
    
    
    label = torch.concat(label_list).view(len(batch),-1)
        
    
    ts_data = torch.concat(ts_list).view(len(batch),24, -1)
    diag_data = torch.concat(diag_list).view(len(batch),-1)
    if len(entity_list)>=1:
        entity_data = torch.concat(entity_list).view(len(batch),512,-1)
    else:
        entity_data = []
    
    return id_list, ts_data, diag_data, entity_data, label, np.array(row_list)

