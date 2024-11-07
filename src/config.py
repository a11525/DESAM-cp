import argparse

def get_args():
    parser = argparse.ArgumentParser()



    #
    parser.add_argument('--task' , type=str, default='train')
    parser.add_argument('--random_seed',type=int, default=0)

    #parser.add_argument('--use_entity', action = 'store_true', default=False)
    parser.add_argument('--all_label', action ='store_true', default =False) 
    #parser.add_argument('--aux_los',action = 'store_true',default=False)
    ## data path ###
    parser.add_argument('--split_folder',type=str, default='split_0')
    parser.add_argument('--data_path',type=str,default='../data/')
    parser.add_argument('--checkpoint_dir', type=str,default = '../checkpoint/')
    

    ### model parameter ###
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    
    parser.add_argument('--save_model',action='store_true', default=False)
    parser.add_argument('--save_name',type=str, default='base')
    parser.add_argument('--last_layer', type=bool, default=False)

    parser.add_argument('--first_last', action='store_true', default = False)
    parser.add_argument('--last_visit', action='store_true', default = False)

    parser.add_argument('--input_dim', type=int , default =768)
    parser.add_argument('--act_func', type=str , default='relu')
    parser.add_argument('--act_init', action='store_true', default = False )
    # self-attention
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layer', type=int, default=6)
    parser.add_argument('--use_last_adm_time',action='store_true',default=False)
    parser.add_argument('--visit_mask', action='store_true', default=False)
    
    ### learning parameter
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                    help='L2 loss on params')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size',type=int, default=64)

    parser.add_argument('--retrain', type=str, default='new')
    parser.add_argument('--patience', type=int , default=10)
    args = parser.parse_args()
    # for jupyter
    #args =parser.parse_args(args=[])
    return args