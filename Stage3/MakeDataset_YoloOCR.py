import yaml 
import json 
import os 
import pandas as pd 
from sklearn.model_selection import train_test_split
import shutil 
from os.path import join 
from tqdm import tqdm 
from PIL import Image, ExifTags

def exif_size(img):
    # Returns exif-corrected PIL size
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

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break



if __name__ == "__main__":
    #============= 1.Read Configs =============
    Yolo_OCR_Configs = yaml.load(open("../configs/yolo-ocr.yaml"), Loader=yaml.FullLoader)
    Dataset_Configs  = yaml.load(open("../configs/data.yaml"), Loader=yaml.FullLoader)
    

    Root_dir    = Yolo_OCR_Configs['YoloOCR-Root']
    img_train   = Yolo_OCR_Configs['YoloOCR-Img-Train']
    label_train = Yolo_OCR_Configs['YoloOCR-Label-Train']
    img_test    = Yolo_OCR_Configs['YoloOCR-Img-Val']
    label_test  = Yolo_OCR_Configs['YoloOCR-Label-Val']

    os.makedirs(img_train, exist_ok = True)
    os.makedirs(label_train, exist_ok = True)
    os.makedirs(img_test, exist_ok = True)
    os.makedirs(label_test, exist_ok = True)

    img_path   = Dataset_Configs['Path-Pres-Img']
    label_path = Dataset_Configs['Path-Pres-Label']
    train_files, test_files, _, _ = train_test_split(os.listdir(label_path), os.listdir(label_path), test_size=0.2, random_state=42)

    
    #============= 2.Create Train Dataset YoloOCR =============
    for idx,json_file in enumerate(tqdm(train_files)):
        with open(os.path.join(label_path, json_file), 'r') as f:
            bill = json.load(f)
        
        # move image to image train 
        image_file = json_file.replace("json", "png")
        shutil.copy(join(img_path,image_file), join(img_train, image_file))
        img = Image.open(join(img_train, image_file))
        W,H = exif_size(img)
        
        with open(join(label_train, json_file.replace("json", "txt")), "w") as f:
            for anno in bill:
                x_min, y_min, x_max, y_max = anno["box"][0], anno["box"][1], anno["box"][2], anno["box"][3]
                x_mid = (x_min/W + x_max/W)/2
                y_mid = (y_min/H + y_max/H)/2
                w_box = (x_max-x_min)/W
                h_box = (y_max-y_min)/H

                ## Text can chu y 
                if "mapping" in anno.keys():
                    f.write(f'1 {x_mid} {y_mid} {w_box} {h_box}\n')
                else:
                    f.write(f'0 {x_mid} {y_mid} {w_box} {h_box}\n')


    #============= 3.Create Valid Dataset YoloOCR =============
    for idx,json_file in enumerate(tqdm(test_files)):
        with open(os.path.join(label_path, json_file), 'r') as f:
            bill = json.load(f)
        
        # move image to image train 
        image_file = json_file.replace("json", "png")
        shutil.copy(join(img_path,image_file), join(img_test, image_file))
        img = Image.open(join(img_test, image_file))
        W,H = exif_size(img)
        
        with open(join(label_test, json_file.replace("json", "txt")), "w") as f:
            for anno in bill:
                x_min, y_min, x_max, y_max = anno["box"][0], anno["box"][1], anno["box"][2], anno["box"][3]
                x_mid = (x_min/W + x_max/W)/2
                y_mid = (y_min/H + y_max/H)/2
                w_box = (x_max-x_min)/W
                h_box = (y_max-y_min)/H

                if "mapping" in anno.keys():
                    f.write(f'1 {x_mid} {y_mid} {w_box} {h_box}\n')
                else:
                    f.write(f'0 {x_mid} {y_mid} {w_box} {h_box}\n')
    

    #============= 4.Create Train.txt + Val.txt =============
    cwd = os.path.abspath(Root_dir)

    with open(join(cwd , 'train.txt'), 'w') as f:
        tmp_dir = os.path.abspath(join(os.getcwd(), img_train))
        for path in os.listdir(tmp_dir):
            f.write(join(tmp_dir, path)+'\n')
            
    with open(join(cwd , 'val.txt'), 'w') as f:
        tmp_dir = os.path.abspath(join(os.getcwd(), img_test))
        for path in os.listdir(tmp_dir):
            f.write(join(tmp_dir, path)+'\n')


    #============= 5. Create Yaml YoloOCR =============
    data = dict(
        train =  join(cwd,'train.txt'),
        val   =  join(cwd,'val.txt'),
        nc    = 2,
        names = ["0", "1"]
    )

    with open(join(cwd,'Yolov5-OCR.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


    f = open(join(cwd,'Yolov5-OCR.yaml'), 'r')
    print('\nyaml:')
    print(f.read())