import json 
import os 
import pandas as pd 
from sklearn.model_selection import train_test_split
import shutil 
from os.path import join 
from tqdm import tqdm 
from PIL import Image 
import PIL 
import yaml

if __name__ == "__main__":
    data_configs    = yaml.load(open("../configs/data.yaml"), Loader=yaml.FullLoader)
    vietocr_configs = yaml.load(open("../configs/vietocr.yaml"), Loader=yaml.FullLoader)

    # Precs Imgs & Labels From Challenges 
    img_path            = data_configs['Path-Pres-Img']
    label_path          = data_configs['Path-Pres-Label']
    train_files, test_files, _, _ = train_test_split(os.listdir(label_path), os.listdir(label_path), test_size=0.2, random_state=42)

    # Output Image & Label
    path_vietocr_root   = vietocr_configs['Root-VietOCR']
    img_path_train_test = vietocr_configs['Imgs-VietOCR']
    label_train         = vietocr_configs['Train-Txt-VietOCR']
    label_test          = vietocr_configs['Test-Txt-VietOCR']
    os.makedirs(img_path_train_test, exist_ok = True)


    # 1. Create Train dataset For RecognizeOCR 
    with open(join(label_train), "w") as file:
        for json_file in tqdm(train_files):
            with open(os.path.join(label_path, json_file), 'r') as f:
                bill = json.load(f)
                
            image_file = json_file.replace("json", "png")

            for idx,anno in enumerate(bill):
                x_min, y_min, x_max, y_max = anno["box"][0], anno["box"][1], anno["box"][2], anno["box"][3]
                img_name =  str(idx) + "_" + image_file
                original = Image.open(join(img_path,image_file))
                img = original.crop((x_min, y_min, x_max, y_max))
                img = img.save(join(img_path_train_test, img_name))
                file.write(f'{img_path_train_test.split("/")[-1]+ "/" + img_name}\t{anno["text"]}\n')
    
    # 2. Create Test dataset For RecognizeOCR
    with open(join(label_test), "w") as file:
        for json_file in tqdm(test_files):
            with open(os.path.join(label_path, json_file), 'r') as f:
                bill = json.load(f)
                
            image_file = json_file.replace("json", "png")

            for idx,anno in enumerate(bill):
                x_min, y_min, x_max, y_max = anno["box"][0], anno["box"][1], anno["box"][2], anno["box"][3]
                img_name =  str(idx) + "_" + image_file
                original = Image.open(join(img_path,image_file))
                img = original.crop((x_min, y_min, x_max, y_max))
                img = img.save(join(img_path_train_test, img_name))
                file.write(f'{img_path_train_test.split("/")[-1]+ "/" + img_name}\t{anno["text"]}\n')
    