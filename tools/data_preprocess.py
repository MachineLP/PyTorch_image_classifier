import os
import sys
import argparse
import numpy as np 
import pandas as pd 
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--data_dir', help='data path', type=str)
parser.add_argument('--n_splits', help='n_splits', type=int)
parser.add_argument('--output_dir', help='output_dir', type=str)
parser.add_argument('--random_state', help='random_state', type=int)
args = parser.parse_args()



if __name__ == '__main__':

    df_data = pd.read_csv(args.data_dir)
    img_path_list = df_data['filepath'].values.tolist()
    label_list = df_data['target'].values.tolist()


    data_label = []
    for per_img_path, per_label in zip( img_path_list, label_list ):          
        data_label.append( [ per_img_path, per_label ] ) 


    train_list = []
    val_list = []
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state) 
    for index, (train_index, val_index) in enumerate(kf.split(data_label)): 
        for i in val_index:
            data_label[i].append(index)
    data_label = np.array( data_label )
    # print (data_label)


    res = DataFrame()
    res['filepath'] = data_label[:,0]
    res['target'] = data_label[:,1]
    res['fold'] = data_label[:,2]
    res[ ['filepath', 'target', 'fold'] ].to_csv(args.output_dir, index=False) 


