import numpy as np, pandas as pd
from glob import glob
import shutil, os
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from os import listdir
from os.path import isfile, join
import argparse
import yaml

import warnings
warnings.filterwarnings("ignore")

def create_args():
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('--Yaml_Data_VAIPE', 
                            type=str, 
                            default="../data/configs/data.yaml", 
                            help='Yaml Data Configs Of Challenge')
    
    my_parser.add_argument('--Yaml_Data_Yolo', 
                            type=str, 
                            default="../data/configs/Yolo5x6_Fold0.yaml", 
                            help='Yaml Yolo5x6 Cross-Fold Configs')
   
    return my_parser.parse_args()
    


if __name__ == "__main__":
    args = create_args()
    Dataset_Configs   = yaml.load(open(args.Yaml_Data_VAIPE), Loader=yaml.FullLoader)
    Yolo_Data_Configs = yaml.load(open(args.Yaml_Data_Yolo), Loader=yaml.FullLoader)
    

    SEED       = Yolo_Data_Configs['SEED']
    n_folds    = Yolo_Data_Configs['NUM_FOLDS']
    fold_num   = Yolo_Data_Configs['FOLD']
    Hyper_Name = Yolo_Data_Configs['YoloPill-HyperParameter']
    train_df   = pd.read_csv(Dataset_Configs['Train-Detection-CSV']).reset_index(drop=True)


    Root_dir    = Yolo_Data_Configs['YoloPill-Root']
    image_train = Yolo_Data_Configs['YoloPill-Img-Train']
    label_train = Yolo_Data_Configs['YoloPill-Label-Train']
    image_val   = Yolo_Data_Configs['YoloPill-Img-Val']
    label_val   = Yolo_Data_Configs['YoloPill-Label-Val']    
    label_dir   = Yolo_Data_Configs['YoloPill-Label-Tmp']


    os.makedirs(label_dir, exist_ok = True)
    os.makedirs(label_train, exist_ok = True)
    os.makedirs(label_val, exist_ok = True)
    os.makedirs(image_train, exist_ok = True)
    os.makedirs(image_val, exist_ok = True)


    # ============= Convert All Labels For All Images & Save To 1 Folder =============
    for img_id in tqdm(list(train_df.image_id.unique())):
        list_img_id = train_df[train_df["image_id"] == img_id].reset_index(drop=True)
        name_file = img_id + ".txt"
        with open(os.path.join(label_dir, name_file), "w") as f:
            for idx in range(len(list_img_id)):
                f.write(str(list_img_id.iloc[idx]["class_id"]))
                f.write(" ")
                f.write('{}'.format(list_img_id.iloc[idx]["x_mid"]))
                f.write(" ")
                f.write('{}'.format(list_img_id.iloc[idx]["y_mid"]))
                f.write(" ")
                f.write('{}'.format(list_img_id.iloc[idx]["w"]))
                f.write(" ")
                f.write('{}'.format(list_img_id.iloc[idx]["h"]))
                f.write("\n")
    

    # ============= Fold Split =============
    gkf  = StratifiedGroupKFold(n_splits = n_folds, random_state=SEED, shuffle=True)
    train_df['fold'] = -1

    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, train_df.class_id.tolist(), groups = train_df.image_id.tolist())):
        train_df.loc[val_idx, 'fold'] = fold

    train_files = []
    val_files   = []

    if fold_num == -1:
        train_files += list(train_df.image_path.unique())
        val_files   += list(train_df.image_path.unique())
    else:
        train_files += list(train_df[train_df.fold!=fold_num].image_path.unique())
        val_files   += list(train_df[train_df.fold==fold_num].image_path.unique())


    # ============= Move File =============
    for file in tqdm(train_files):
        shutil.copy(file, image_train)
        filename = file.split('/')[-1].split('.')[0]
        shutil.copy(os.path.join(label_dir, filename+'.txt'), label_train)
        
    for file in tqdm(val_files):
        shutil.copy(file, image_val)
        filename = file.split('/')[-1].split('.')[0]
        shutil.copy(os.path.join(label_dir, filename+'.txt'), label_val)

    
    # ============== Class Id =============
    class_ids, class_names = list(zip(*set(zip(train_df.class_id, train_df.class_name))))
    classes = list(np.array(class_names)[np.argsort(class_ids)])
    classes = list(map(lambda x: str(x), classes))


    # ============= Make train.txt + val.txt =============
    cwd = os.path.abspath(Root_dir)
    with open(join(cwd , 'train.txt'), 'w') as f:
        tmp_dir = os.path.abspath(join(os.getcwd(), image_train))
        for path in os.listdir(tmp_dir):
            f.write(join(tmp_dir, path)+'\n')
                
    with open(join(cwd , 'val.txt'), 'w') as f:
        tmp_dir = os.path.abspath(join(os.getcwd(), image_val))
        for path in os.listdir(tmp_dir):
            f.write(join(tmp_dir, path)+'\n')

    data= dict(
            train =  join(cwd, 'train.txt') ,
            val   =  join(cwd, 'val.txt' ),
            nc    = len(classes),
            names = list(classes)
        )

    with open(join(cwd,'VAIPE-YOLOV5x6.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    f = open(join(cwd,'VAIPE-YOLOV5x6.yaml'), 'r')
    print('\nyaml:')
    print(f.read())


    # ============= Make HyperParameter =============
    hyperameter_default = dict(lr0= 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
                        lrf= 0.01,  # final OneCycleLR learning rate (lr0 * lrf)
                        momentum= 0.937,  # SGD momentum/Adam beta1
                        weight_decay= 0.0005,  # optimizer weight decay 5e-4
                        warmup_epochs= 3.0,  # warmup epochs (fractions ok)
                        warmup_momentum= 0.8,  # warmup initial momentum
                        warmup_bias_lr= 0.1,  # warmup initial bias lr
                        box= 0.05,  ###
                        cls= 0.5,  # cls loss gain
                        cls_pw= 1.0,  # cls BCELoss positive_weight
                        obj= 1.0,  # obj loss gain (scale with pixels)
                        obj_pw= 1.0,  # obj BCELoss positive_weight
                        iou_t= 0.30,  ###
                        anchor_t= 4.0,  # anchor-multiple threshold
                        # anchors: 3  # anchors per output layer (0 to ignore)
                        fl_gamma= 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
                        hsv_h= 0.015,  # image HSV-Hue augmentation (fraction)
                        hsv_s= 0.7,  ##
                        hsv_v= 0.0,  ##
                        degrees= 0.0,  # image rotation (+/- deg)
                        translate= 0.1,  # image translation (+/- fraction)
                        scale= 0.5,  # image scale (+/- gain)
                        shear= 0.0,  # image shear (+/- deg)
                        perspective= 0.0,  # image perspective (+/- fraction), range 0-0.001
                        flipud= 0.0,  # image flip up-down (probability)
                        fliplr= 0.5,  # image flip left-right (probability)
                        mosaic= 0.3,  ##
                        mixup= 0.0,  # image mixup (probability)
                        copy_paste= 0.0  # segment copy-paste (probability)
                      )
    
    with open(join( cwd, Hyper_Name), 'w') as outfile:
        yaml.dump(hyperameter_default, outfile, default_flow_style=False)
        
    print(f'Saved file {join(cwd , Hyper_Name)}')