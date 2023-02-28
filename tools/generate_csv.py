
from pathlib import Path
import os
from pandas.core.frame import DataFrame


video_path = "./data/download_images/"
all_labels_path = os.listdir(video_path)

all_img_path_list = []
content_label_list = []
fold_list = []
for per_label_path in all_labels_path:
    if 'csv' in per_label_path:
        continue
    per_all_img_path = os.path.join(video_path, per_label_path)
    all_image_path = os.listdir(per_all_img_path)
    for per_image_path in all_image_path:
        per_img_path = os.path.join(video_path, per_label_path, per_image_path)
        all_img_path_list.append( per_img_path )
        content_label_list.append( per_label_path.split('ads')[0][:-1] )
        fold_list.append( 0 )
        



res = DataFrame()
res['filepath'] = list( all_img_path_list )
res['target'] = list( content_label_list )
res['fold'] = list( fold_list )
res[ ['filepath', 'target', 'fold'] ].to_csv('./data/data.csv', index=False) 




