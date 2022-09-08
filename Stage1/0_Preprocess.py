import numpy as np
import pandas as pd
from glob import glob
import shutil, os
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from tqdm.notebook import tqdm
import seaborn as sns
import json
import cv2 
from tqdm import tqdm 

import yaml
from PIL import Image, ExifTags

def exif_size(img):
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass
    return s


def get_path(image_id):
    return os.path.join(Root_Path_Pill_Img, image_id + ".jpg")


if __name__ == "__main__":
    configs = yaml.load(open("../configs/data.yaml"), Loader=yaml.FullLoader)
    
    Root_Path_Pill              = configs['Path-Pill']
    Root_Path_Pill_Img          = configs['Path-Pill-Img']
    Root_Path_Pill_Label        = configs['Path-Pill-Label']

    Root_Path_MedicalBill       = configs['Path-Pres']
    Root_Path_MedicalBill_Img   = configs['Path-Pres-Img']
    Root_Path_MedicalBill_Label = configs['Path-Pres-Label']

    Train_Detection_CSV  = configs['Train-Detection-CSV']

    image_id     = []
    class_id     = []
    class_name   = [] 
    xmin_boxes   = []
    ymin_boxes   = []
    xmax_boxes   = []
    ymax_boxes   = []
    w_boxes      = []
    h_boxes      = []
    heights      = []
    widths       = []

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    #1. Read Info Label From Dataset 
    for json_file in tqdm(os.listdir(Root_Path_Pill_Label)):
        img = Image.open(os.path.join(Root_Path_Pill_Img, json_file.replace(".json", ".jpg")))
        W,H = exif_size(img)
    
        with open(os.path.join(Root_Path_Pill_Label, json_file), 'r') as f:
            dataset = json.load(f)
            
        for item in dataset:
            image_id.append(json_file.replace(".json", ""))
            class_id.append(item['label'])
            class_name.append(str(item['label']))
            
            xmin_boxes.append(item['x'])
            ymin_boxes.append(item['y'])
            
            xmax_boxes.append(item['x'] + item['w'])
            ymax_boxes.append(item['y'] + item['h'])
            
            w_boxes.append(item['w'])
            h_boxes.append(item['h'])
            
            heights.append(H)
            widths.append(W)
    
    # 2. Create CSV File With All Info
    train_df = pd.DataFrame({"image_id":image_id,
                            "class_id":class_id,
                            "class_name":class_name,
                            "x_min"  :xmin_boxes,
                            "y_min"  :ymin_boxes,
                            "x_max"  :xmax_boxes,
                            "y_max"  :ymax_boxes,
                            "w_box"  :w_boxes,
                            "h_box"  :h_boxes,
                            "width"  :widths,
                            "height" :heights})

    train_df = train_df.reset_index(drop=True)
    train_df['image_path'] = train_df["image_id"].apply(get_path)

    train_df['x_mid'] = train_df.apply(lambda row: (row.x_min/row.width +  (row.x_min/row.width + row.w_box/row.width))/2 , axis =1)
    train_df['y_mid'] = train_df.apply(lambda row: (row.y_min/row.height + (row.y_min/row.height+ row.h_box/row.height))/2, axis =1)

    train_df['w'] = train_df.apply(lambda row: row.w_box/row.width, axis =1)
    train_df['h'] = train_df.apply(lambda row: row.h_box/row.height, axis =1)

    train_df['area'] = train_df['w']*train_df['h']
    
    if not os.path.isfile(Train_Detection_CSV): 
        train_df.to_csv(Train_Detection_CSV, index=False)

    print(train_df.head(5))
