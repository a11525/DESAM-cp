from config import get_args
import torch
import numpy as np
import pickle
import utils,  models, dataset, trainer
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import sys
import pandas as pd
import pdb
from sklearn.metrics import roc_auc_score, precision_score , recall_score,accuracy_score
import warnings
import time
warnings.filterwarnings('ignore')


args = get_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
args.device = device

print('Run model with {} device'.format(device))
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)


if __name__ == "__main__":

    args.first_last=True

    
    traindata = dataset.EXTRACTDataset(args, 'train')
    
    
    validdata =dataset.EXTRACTDataset(args,'dev')
    

    testdata = dataset.EXTRACTDataset(args, 'test')
    
    train_loader = DataLoader(traindata, batch_size=args.batch_size,num_workers=0,
                            shuffle=True)
    valid_loader = DataLoader(validdata,batch_size=args.batch_size,num_workers=0,
                            shuffle=False)
    test_loader = DataLoader(testdata,batch_size=args.batch_size,num_workers=0,
                            shuffle=False)
    
    args.TS_dimension_size = traindata.ts_data.shape[-1] # 104 using mimic-extract
    args.entity_dimension_size = 1024 # biobert dimension
    args.doc_dimension_size = 256 # tf-idf
    args.label_dim= traindata.label_dim()

    model = models.EntityAttentionModel(args)
    patience_cnt = 0
    patience = args.patience

    if args.task =='train':
        model.to(device)
        criterion = torch.nn.BCELoss(reduction='sum')
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()), lr=args.learning_rate,
                                weight_decay=args.weight_decay)
        if args.save_model:

            if os.path.isdir(args.checkpoint_dir + args.save_name) == False:
                os.mkdir(args.checkpoint_dir + args.save_name)
                
        if args.retrain=='keep':
            model.load_state_dict(torch.load(args.checkpoint_dir + args.save_name +'/'+
                        args.save_name+'.pth'), strict=False)
            model.to(device)
        valid_loss, pred_list, true_list = trainer.eval_step(model,criterion, valid_loader, args)
        
        auc_list, aupr_list = utils.multi_class_cal_metric(pred_list, true_list)
        print('Before Valid metric ( auc| aupr) : mort_hosp / mort_icu/  los_3/ los_7]'
        ' : [ ({:.4f}| {:.4f}) / ({:.4f}| {:.4f}) / ({:.4f}| {:.4f}) / ({:.4f}| {:.4f})]'.format(\
        auc_list[0], aupr_list[0],auc_list[1], aupr_list[1],auc_list[2], aupr_list[2], auc_list[3], aupr_list[3]))
        
        print('\n')
        best_loss=valid_loss

        for k in range(args.epochs):
            s_time= time.time()
            
            train_loss = trainer.train_step(model, criterion, optimizer, train_loader, args)
            valid_loss, pred_list, true_list = trainer.eval_step(model,criterion, valid_loader, args)

            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print("Process epoch: [{}/{}] Train Loss: [{:.4f}]  Valid loss: [{:.4f}]  with {:.1f} min\n"\
                .format(k+1, args.epochs, train_loss, valid_loss, (time.time()-s_time)/60))
            
            

            if best_loss > valid_loss:
                patience_cnt=0

                best_loss = valid_loss
                best_epoch= k
                best_train_loss=train_loss

                if args.save_model:
                    torch.save(model.state_dict(), args.checkpoint_dir + args.save_name +'/'+
                        args.save_name+'.pth')
            else:
                patience_cnt += 1

            if patience_cnt == patience:
                break
        print('\n')
        print("Best valid epoch : [{}]   train loss : [{:.4f}]  valid loss : [{:.4f}]" \
            .format(best_epoch+1, best_train_loss, best_loss))
        print('\n')

        model.load_state_dict(torch.load(args.checkpoint_dir + args.save_name +'/'+
                        args.save_name+'.pth'), strict=False)

        valid_loss, val_pred_list, val_true_list = trainer.eval_step(model,criterion, valid_loader, args)
        test_loss, test_pred_list, test_true_list = trainer.eval_step(model,criterion, test_loader, args)

        print("best Valid loss: [{:.6f}]  Test loss: [{:.6f}]"\
                .format( valid_loss,test_loss))

        v_auc_list, v_aupr_list = utils.multi_class_cal_metric(val_pred_list, val_true_list)
        print('Best Valid metric ( auc| aupr) : mort_hosp / mort_icu/  los_3/ los_7]'
        ' : [ ({:.4f}| {:.4f}) / ({:.4f}| {:.4f}) / ({:.4f}| {:.4f}) / ({:.4f}| {:.4f})]'.format(\
        v_auc_list[0], v_aupr_list[0],v_auc_list[1], v_aupr_list[1],v_auc_list[2], v_aupr_list[2], v_auc_list[3], v_aupr_list[3]))
        print('val predict death num : {} / {}  :'.format(val_pred_list.round().sum(),val_true_list.round().sum()))

        t_auc_list, t_aupr_list = utils.multi_class_cal_metric(test_pred_list, test_true_list)
        print('Best test metric ( auc| aupr) : mort_hosp / mort_icu/  los_3/ los_7]'
        ' : [ ({:.4f}| {:.4f}) / ({:.4f}| {:.4f}) / ({:.4f}| {:.4f}) / ({:.4f}| {:.4f})]'.format(\
        t_auc_list[0], t_aupr_list[0],t_auc_list[1], t_aupr_list[1],t_auc_list[2], t_aupr_list[2], t_auc_list[3], t_aupr_list[3]))
        print('test predict death num : {} / {}  :'.format(test_pred_list.round().sum(),test_true_list.round().sum()))

    elif args.task =='test':
        model.load_state_dict(torch.load(args.checkpoint_dir + args.save_name +'/'+
                        args.save_name+'.pth'), strict=False)
        model.to(device)
        criterion = torch.nn.BCELoss(reduction='sum')
        
        valid_loss, val_pred_list, val_true_list = trainer.eval_step(model,criterion, valid_loader, args)
        test_loss, test_pred_list, test_true_list = trainer.eval_step(model,criterion, test_loader, args)

        
        v_auc_list, v_aupr_list = utils.multi_class_cal_metric(val_pred_list, val_true_list)
        print('Best Valid metric ( auc| aupr) : mort_hosp / mort_icu/  los_3/ los_7]'
        ' : [ ({:.4f}| {:.4f}) / ({:.4f}| {:.4f}) / ({:.4f}| {:.4f}) / ({:.4f}| {:.4f})]'.format(\
        v_auc_list[0], v_aupr_list[0],v_auc_list[1], v_aupr_list[1],v_auc_list[2], v_aupr_list[2], v_auc_list[3], v_aupr_list[3]))
        print('val predict death num : {} / {}  :'.format(val_pred_list.round().sum(),val_true_list.round().sum()))

        t_auc_list, t_aupr_list = utils.multi_class_cal_metric(test_pred_list, test_true_list)
        print('Best test metric ( auc| aupr) : mort_hosp / mort_icu/  los_3/ los_7]'
        ' : [ ({:.4f}| {:.4f}) / ({:.4f}| {:.4f}) / ({:.4f}| {:.4f}) / ({:.4f}| {:.4f})]'.format(\
        t_auc_list[0], t_aupr_list[0],t_auc_list[1], t_aupr_list[1],t_auc_list[2], t_aupr_list[2], t_auc_list[3], t_aupr_list[3]))
        print('test predict death num : {} / {}  :'.format(test_pred_list.round().sum(),test_true_list.round().sum()))

    elif args.task =='predict':
        
        model.load_state_dict(torch.load(args.checkpoint_dir + args.save_name +'/'+
                        args.save_name+'.pth'), strict=False)
        model.to(device)

        all_pred = trainer.predict_step(model, train_loader, args)
        test_pred = trainer.predict_step(model, test_loader, args)
        pd.to_pickle(all_pred, args.checkpoint_dir + args.save_name + '/train_predict_sample.pkl')
        pd.to_pickle(test_pred, args.checkpoint_dir + args.save_name + '/test_predict_sample.pkl')
    