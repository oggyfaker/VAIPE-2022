import matplotlib.pyplot as plt
import pandas as pd 
import json 
import cv2 
from os.path import join 
import os
import torch 
from bbox.utils import coco2yolo, coco2voc, voc2yolo, voc2coco, yolo2coco
from PIL import Image
from tqdm import tqdm 

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from IPython.display import display

import warnings
from IPython.display import display
import yaml 
import argparse

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

def create_args():
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('--Configs_Recore', 
                            type=str, 
                            default="../data/configs/data.yaml", 
                            help='Configs recore csv dataset')

    return my_parser.parse_args()

"""
Yolov5 Text Detection
Output: Voc Format 
"""
def load_image_pres(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
    return img 

def Load_Model_YoloText(ckpt_path, conf=0.25, iou=0.50):
    model = torch.hub.load('./yolov5',
                           'custom',
                           path=ckpt_path,
                           source='local',
                           force_reload=True)  # local repo
    model.conf = conf  # NMS confidence threshold
    model.iou  = iou  # NMS IoU threshold
    return model
    
def predict_text_bbox(model, img, size=768, augment=False):
    height, width = img.shape[:2]
    results = model(img, size=size, augment=augment)
    preds   = results.pandas().xyxy[0]
    bboxes  = preds[['xmin','ymin','xmax','ymax']].values
    if len(bboxes):
        confs   = preds.confidence.values
        classes = preds.name.values
        return bboxes, confs, classes
    else:
        return [],[],[]

"""
Filter Bounding Box By Class 
Use for Filter Box of YoloOCR prediction (Just only using box contain name of medical)
"""
def filter_box_class(bboxes,confs,labels, label_choice=1):
    if len(bboxes) == 0:
        return [], [], []
    else:
        filtered_bboxes, filtered_confs, filtered_labels = [], [], []
        for bbox, conf,label in zip(bboxes, confs,labels):
            if int(label) == label_choice:
                bbox = [int(tmp) for tmp in list(bbox)]
                filtered_bboxes.append(bbox)
                filtered_confs.append(float(conf))
                filtered_labels.append(int(label))
        return filtered_bboxes, filtered_confs, filtered_labels


    
"""
VietOCR Recognize Text 
"""
def VietOCR_Recognize_String(model, boxes, img_path, debug=False):
    string_predictions = []
    img = Image.open(img_path)
    
    if len(boxes) > 0:
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            img_crop = img.crop((xmin, ymin, xmax, ymax))
            string = model.predict(img_crop)
            string_predictions.append(string)

            if debug:
                print(string)
                display(img_crop)

        if len(string_predictions)>0: return string_predictions
        else: return [] 
    else:
        print("No Bouding Box Of Text !!")
        return []



"""
Format string labels 
"""
import gensim
import re

def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))

def Lowercase(inputString):
    return inputString.lower()

def Uppercase(inputString):
    return inputString.upper()

def remove_num_head(inputString):
    if ")" in inputString.split(" ")[0] and has_numbers(inputString.split(" ")[0]):
        return " ".join(inputString.split(" ")[1:])
    else:
        return inputString

def remove_num_in_string(inputString):
    list_string = inputString.split(" ")
    new_string  = [] 
    for word in list_string:
        if has_numbers(word):
            continue
        else:
            new_string.append(word)
    return " ".join(new_string)

def clear_string(string, remove_number_head=True, remove_number_in_string=True, lowercase=False):
    if remove_number_head:
        string = remove_num_head(string)
    if remove_number_in_string:
        string = string.replace("-", " ")
        string = remove_num_in_string(string)    
    if lowercase:
        string = Lowercase(string)
    else:
        string = Uppercase(string)
    return string.strip().replace("+", " ")



"""
Find Label By Text 
"""
# !git clone -q https://github.com/luozhouyang/python-string-similarity.git
import sys
sys.path.append("python-string-similarity")
from strsimpy.jaro_winkler import JaroWinkler

def Create_Dict_GroundTruth(csv_file, remove_number_head, remove_number_in_string, lowercase):
    ground_truth_dict = {} 
    list_names = csv_file["names"].unique()
    for name in list_names:
        tmp = csv_file[csv_file["names"]==name].reset_index(drop=True)
        list_label = list(tmp["class_id"].unique())
        
        if len(list_label) != 0:
            name = clear_string(name, 
                                remove_number_head=remove_number_head, 
                                remove_number_in_string=remove_number_in_string, 
                                lowercase=lowercase)
            ground_truth_dict[name] = list_label
        else:
            continue
    return ground_truth_dict

def simlarity_score(str1, str2):
    jarowinkler = JaroWinkler()
    score = jarowinkler.similarity(str1, str2)
    return score 

import heapq
from operator import itemgetter
def find_numclass_by_name(GroundTruth_Dict, List_name, TopK=1,debug=False):
    Class_In_Image = [] 
    Query = {}
    
    for name in List_name:
        for truth_text in list(GroundTruth_Dict.keys()):
            Query[truth_text] = simlarity_score(truth_text, name)    
        
        result = heapq.nlargest(TopK, Query.items(), key=itemgetter(1))
        for text,score in result:
            Class_In_Image.append(GroundTruth_Dict[text])
            if debug:
                print(f'[{name}] x [{text}]  --> Score Simlarity: ', score)
    return [item for sub in Class_In_Image for item in sub]


def find_not_class(lst1,lst2):
    return [item for item in lst2 if item not in lst1]


def find_lost_class(df, image_id, list_class_from_ocr):
    """
    Function find lost class
    Input:
        df: Dataframe of submission 
        image_id: The id of image pill for consider 
        list_class_from_ocr: all class from OCR  
    Output: 
        fail_class: Fail class (contain in Pill Detection but not in OCR)
        lost_class: Missing class (contain in OCR but not in Pill Detection) 
        now_class: Class of Pill Detecion
    """
    df = df[df["image_name"] == image_id].reset_index(drop=True)
    now_class  = list(df['class_id'].unique())
    lost_class = [tmp for tmp in list_class_from_ocr if tmp not in list(df['class_id'].unique()) and tmp != 107]
    fail_class = [tmp for tmp in list(df['class_id'].unique()) if tmp not in list_class_from_ocr and tmp != 107]
    return now_class, lost_class, fail_class



if __name__ == "__main__":
    args = create_args()
    RescoreConfigs = yaml.load(open(args.Configs_Recore), Loader=yaml.FullLoader)

    #-------------------------------------------------
    PUBLIC_TEST   = RescoreConfigs['TEST_PATH']
    MAP_JSON_FILE = RescoreConfigs['JSON_TEST']
    PRESC_IMAGES  = RescoreConfigs['PRESC_IMG_TEST']
    PILL_IMAGES   = RescoreConfigs['PILL_IMG_TEST']
    RESULT_PILL   = RescoreConfigs['CSV_Pill_Yolo_Result'] # Result from Yolo Ensemble 

    TRAIN_CSV     = RescoreConfigs['TRAIN_CSV']
    TRAIN_PRESC_LABEL = RescoreConfigs['TRAIN_PRESC_LABEL']

    #-------------------------------------------------
    CKPT_PATH_TEXT = RescoreConfigs['WEIGHTS_YOLO_OCR']
    IMGT  = RescoreConfigs['IMGT']
    AUGT  = RescoreConfigs['AUGT']
    CONFT = RescoreConfigs['CONFT']
    IOUT  = RescoreConfigs['IOUT']

    Model_Text = Load_Model_YoloText(ckpt_path=CKPT_PATH_TEXT, conf=CONFT, iou=IOUT)
    CLEAR_HEAD = True
    CLEAR_NUM  = False 
    LOWER      = False

    #-------------------------------------------------
    config = Cfg.load_config_from_file(RescoreConfigs['CONFIGS_TRANSFORMER'])
    config['weights'] = RescoreConfigs['WEIGHTS_TRANSFORMER']
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=False
    detector = Predictor(config)



    #-------------------------- Start Rescore Label By Text Presc --------------------------
    # 1.1.Read result submission & Ground Truth Train - Read Ground Truth Label and Text
    result = pd.read_csv(RESULT_PILL)
    train_csv = pd.read_csv(TRAIN_CSV)
    ground_truth_dict = Create_Dict_GroundTruth(train_csv, 
                                                remove_number_head=CLEAR_HEAD, 
                                                remove_number_in_string=CLEAR_NUM, 
                                                lowercase=LOWER) #{Text: Label_num}

    # 2.1.Read Mapping File 
    with open(MAP_JSON_FILE, 'r') as f:
        pill_pres_map = json.load(f)    


    # 2.2.Convert {Pill Image Id: Pres ID} 
    Map_Pill_Pres = {} 
    for tmp in pill_pres_map:
        for pill_id in tmp["pill"]:
            pill_id = pill_id.replace(".json", "")
            pres_id = tmp['pres'].replace(".json", "")
            Map_Pill_Pres[pill_id] = pres_id


    # 3. Replace fucking class == 107 
    total_class = len(list(result['image_name']))
    count_class_change = 0     
    for pill_img in tqdm(list(result['image_name'].unique())):
        pill_id = pill_img.replace(".jpg", "")
        pres_id = Map_Pill_Pres[pill_id]
        pres_img = load_image_pres(join(PRESC_IMAGES,pres_id+".png"))
        
        # Predict Text Box 
        box_text, score_text, class_text = predict_text_bbox(Model_Text, pres_img, size=IMGT, augment=AUGT)
        box_text, score_text, class_text = filter_box_class(box_text, score_text, class_text, label_choice=1)
        
        # Text Recognize 
        detected_strings = VietOCR_Recognize_String(model=detector, boxes=box_text, img_path=join(PRESC_IMAGES,pres_id+".png"), debug=False)
        
        # Format Text 
        if len(detected_strings) != 0:
            detected_strings =   [clear_string(string, remove_number_head=CLEAR_HEAD, remove_number_in_string=CLEAR_NUM, lowercase=LOWER) for string in detected_strings]
        else:
            print("---------------")
            print("--> No Found String: ", pres_id)
            display(Image.open(join(PRESC_IMAGES,pres_id+".png")))
            _ = VietOCR_Recognize_String(model=detector, boxes=box_text, img_path=join(PRESC_IMAGES,pres_id+".png"), debug=False)
            print("--------------")
            break 
        
        # list class from presc image 
        class_should_be_haved = find_numclass_by_name(ground_truth_dict, detected_strings, TopK=1, debug=False)
        # Find lost class
        now_class, lost_class, fail_class= find_lost_class(df=result, image_id=pill_img, list_class_from_ocr=class_should_be_haved)
        
        
        # Recorrect class by class_should_be_haved
        if len(lost_class)==1  and len(fail_class)==1:
            result.loc[(result["image_name"]==pill_img) & (result["class_id"]==fail_class[0]), 'class_id'] = lost_class[0]
            count_class_change += 1
    
        elif len(fail_class)!=0:
            count_class_change += len(fail_class)
            for fail in fail_class:
                result.loc[(result["image_name"]==pill_img) & (result["class_id"]==fail), 'class_id'] = 107

    print("% Class changed --> ", (count_class_change/total_class)*100)

    result.to_csv(RescoreConfigs['CSV_Rescore_Result'], index=False)

    result_path = os.path.abspath(RescoreConfigs['CSV_Rescore_Result'])
    print(f'Result saved in --> {result_path}')

    print(result.head(5))