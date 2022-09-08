import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()
import pandas as pd
import os
from os.path import join
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import shutil
import sys
import torch
from PIL import Image
import ast
from pylab import rcParams
import argparse
from itertools import chain

from bbox.utils import coco2yolo, coco2voc, voc2yolo, voc2coco, yolo2coco
from bbox.utils import draw_bboxes, load_image
from bbox.utils import clip_bbox, str2annot, annot2str
import yaml

def create_args():
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('--Configs_Inference', 
                            type=str, 
                            default="../data/configs/data.yaml", 
                            help='Yaml Data Configs Of Challenge')
    
    return my_parser.parse_args()
    


def load_model(ckpt_path, conf=0.25, iou=0.50):
    model = torch.hub.load('./yolov5',
                           'custom',
                           path=ckpt_path,
                           source='local',
                           force_reload=True)  # local repo
    model.conf = conf  # NMS confidence threshold
    model.iou  = iou  # NMS IoU threshold
    return model


def predict(model, img, size=768, augment=False):
    height, width = img.shape[:2]
    results = model(img, size=size, augment=augment)  # custom inference size
    preds   = results.pandas().xyxy[0]
    bboxes  = preds[['xmin','ymin','xmax','ymax']].values
    if len(bboxes):
        bboxes  = voc2coco(bboxes,height,width).astype(int)
        confs   = preds.confidence.values
        classes = preds.name.values
        classes = np.array([int(tmp) for tmp in classes]) 
        return bboxes, confs,classes
    else:
        return [],[],[]


def format_prediction(bboxes, confs):
    annot = ''
    if len(bboxes)>0:
        for idx in range(len(bboxes)):
            xmin, ymin, w, h = bboxes[idx]
            conf             = confs[idx]
            annot += f'{conf} {xmin} {ymin} {w} {h}'
            annot +=' '
        annot = annot.strip(' ')
    return annot


def show_img(img, bboxes, bbox_format='yolo', bbox_colors = None):
    names  = ['label']*len(bboxes)
    labels = [0]*len(bboxes)
    colors = [(0, 255 ,0)]*len(bboxes)
    img    = draw_bboxes(img = img,
                           bboxes = bboxes,
                           classes = names,
                           class_ids = labels,
                           class_name = True,
                           colors = colors if bbox_colors is None else bbox_colors,
                           bbox_format = bbox_format,
                           line_thickness = 2)
    return Image.fromarray(img)


# git clone https://github.com/ZFTurbo/Weighted-Boxes-Fusion.git
sys.path.append("Weighted-Boxes-Fusion")
from ensemble_boxes import *
import numpy as np 
"""
Format for run Ensemble Boxes: COCO (xmin, ymin, w, h)
"""
def run_nms(bboxes, confs,classs, image_size, iou_thr=0.50, skip_box_thr=0.0001, weights=None):
    boxes =  [bbox/(image_size-1) for bbox in bboxes]
    scores = [conf for conf in confs]    
    labels = [item for item in classs]
    boxes, scores, labels = nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def run_nmw(bboxes, confs,classs, image_size, iou_thr=0.50, skip_box_thr=0.0001, weights=None):
    boxes =  [bbox/(image_size-1) for bbox in bboxes]
    scores = [conf for conf in confs]    
    labels = [item for item in classs]
    boxes, scores, labels = non_maximum_weighted(boxes, scores, labels, weights=weights, iou_thr=iou_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def run_soft_nms(bboxes, confs,classs, image_size, iou_thr=0.50, skip_box_thr=0.0001,sigma=0.1, weights=None):
    boxes =  [bbox/(image_size-1) for bbox in bboxes]
    scores = [conf for conf in confs]    
    labels = [item for item in classs]
    boxes, scores, labels = soft_nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def run_wbf(bboxes, confs,classs, image_size, iou_thr=0.50, skip_box_thr=0.0001, weights=None, conf_type='box_and_model_avg'):
    boxes =  [bbox/(image_size-1) for bbox in bboxes]
    scores = [conf for conf in confs]    
    labels = [item for item in classs]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def run_wbf_exp(bboxes, confs,classs, image_size, iou_thr=0.50, skip_box_thr=0.0001, weights=None, conf_type='box_and_model_avg'):
    boxes =  [bbox/(image_size-1) for bbox in bboxes]
    scores = [conf for conf in confs]    
    labels = [item for item in classs]
    boxes, scores, labels =  weighted_boxes_fusion_experimental(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels





if __name__ == "__main__":
    args = create_args()
    InferConfigs = yaml.load(open(args.Configs_Inference), Loader=yaml.FullLoader)
    
    
    #--------------------------------------------------------------------------
    TEST_PATH = InferConfigs["TEST_PATH"]

    CKPT_PATH_1800_all = InferConfigs['WEIGHTS_1800']
    CONF_1800_all      = InferConfigs['CONF_1800']
    IOU_1800_all       = InferConfigs['IOU_1800']

    CKPT_PATH_2432_all = InferConfigs['WEIGHTS_2432']
    CONF_2432_all      = InferConfigs['CONF_2432']
    IOU_2432_all       = InferConfigs['IOU_2432']

    CKPT_PATH_3008_all = InferConfigs['WEIGHTS_3000']
    CONF_3008_all      = InferConfigs['CONF_3000']
    IOU_3008_all       = InferConfigs['IOU_3000']

    CKPT_PATH1 = InferConfigs['WEIGHTS_1280_0']
    CONF1      = InferConfigs['CONF_1280_0']
    IOU1       = InferConfigs['IOU_1280_0']

    CKPT_PATH2 = InferConfigs['WEIGHTS_1280_1']
    CONF2      = InferConfigs['CONF_1280_1']
    IOU2       = InferConfigs['IOU_1280_1']

    CKPT_PATH3 = InferConfigs['WEIGHTS_1280_2']
    CONF3      = InferConfigs['CONF_1280_2']
    IOU3       = InferConfigs['IOU_1280_2']
    
    CKPT_PATH4 = InferConfigs['WEIGHTS_1280_3']
    CONF4      = InferConfigs['CONF_1280_3']
    IOU4       = InferConfigs['IOU_1280_3']

    CKPT_PATH5 = InferConfigs['WEIGHTS_1280_4']
    CONF5      = InferConfigs['CONF_1280_4']
    IOU5       = InferConfigs['IOU_1280_4']
    
    #--------------------------------------------------------------------------
    model1 = load_model(CKPT_PATH1, conf=CONF1, iou=IOU1) 
    model2 = load_model(CKPT_PATH2, conf=CONF2, iou=IOU2)
    model3 = load_model(CKPT_PATH3, conf=CONF3, iou=IOU3)
    model4 = load_model(CKPT_PATH4, conf=CONF4, iou=IOU4)
    model5 = load_model(CKPT_PATH5, conf=CONF5, iou=IOU5)

    model_1800_all = load_model(CKPT_PATH_1800_all, conf=CONF_1800_all, iou=IOU_1800_all) 
    model_2432_all = load_model(CKPT_PATH_2432_all, conf=CONF_2432_all, iou=IOU_2432_all) 
    model_3008_all = load_model(CKPT_PATH_3008_all, conf=CONF_3008_all, iou=IOU_3008_all) 
    
    #--------------------------------------------------------------------------
    # Method          = InferConfigs['METHOD_ENSEMBLE'] # NMS,Soft-NMS,NMW or WBF or WBF_Exp
    # WBF_conf_type   = InferConfigs['WBF_conf_type'] # box_and_model_avg + avg + max + absent_model_aware_avg
    # IMAGE_SIZE_3008 = InferConfigs['IMAGE_SIZE_3000'] # best with 3328
    # IMAGE_SIZE_2432 = InferConfigs['IMAGE_SIZE_2432'] # best with 3000
    # IMAGE_SIZE_1800 = InferConfigs['IMAGE_SIZE_1800'] # best with 2560
    # IMG_SIZE        = InferConfigs['IMAGE_SIZE_1280'] #best with 1664
    # AUGMENT         = InferConfigs['AUGMENT']

    # iou_thr = InferConfigs['iou_thr']
    # skip_box_thr = InferConfigs['skip_box_thr']
    # sigma = InferConfigs['sigma']
    # weights = InferConfigs['weights']

    Method = "NMW" # NMS,Soft-NMS,NMW or WBF or WBF_Exp
    WBF_conf_type = "max" # box_and_model_avg + avg + max + absent_model_aware_avg
    IMAGE_SIZE_3008 = 3328 # best with 3264
    IMAGE_SIZE_2432 = 3000 # best with 3000
    IMAGE_SIZE_1800 = 2560 # best with 2560
    IMG_SIZE  = 1664 #best with 1664
    AUGMENT   = True

    iou_thr = 0.7
    skip_box_thr = 0.001
    sigma = 0.01
    weights = None

    #--------------------------------------------------------------------------
    image_name = [] 
    class_id = [] 
    confidence = [] 
    x_min = [] 
    x_max = [] 
    y_min = [] 
    y_max = [] 

    for idx,file_img in enumerate(tqdm(os.listdir(TEST_PATH))):
        img = cv2.imread(join(TEST_PATH, file_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
        height, width = img.shape[:2]
        IMG_SIZE_ENSEMBLE = max(height, width) 

        bboxes1, scores1,bbclasses1  = predict(model1, img, size=IMG_SIZE, augment=AUGMENT)
        bboxes2, scores2,bbclasses2  = predict(model2, img, size=IMG_SIZE, augment=AUGMENT)
        bboxes3, scores3,bbclasses3  = predict(model3, img, size=IMG_SIZE, augment=AUGMENT)
        bboxes4, scores4,bbclasses4  = predict(model4, img, size=IMG_SIZE, augment=AUGMENT)
        bboxes5, scores5,bbclasses5  = predict(model5, img, size=IMG_SIZE, augment=AUGMENT)
        bboxes6, scores6,bbclasses6  = predict(model_1800_all, img, size=IMAGE_SIZE_1800, augment=AUGMENT)
        bboxes7, scores7,bbclasses7  = predict(model_2432_all, img, size=IMAGE_SIZE_2432, augment=AUGMENT)
        bboxes8, scores8,bbclasses8  = predict(model_3008_all, img, size=IMAGE_SIZE_3008, augment=AUGMENT)


        """
        box format: Coco Format 
        """
        boxes_list=[]
        scores_list=[]
        labels_list=[]
        
        if len(bboxes1)>0:
            boxes_list.append(bboxes1)
            scores_list.append(scores1)
            labels_list.append(bbclasses1) 
            bboxes1[:, 2] = bboxes1[:, 2] + bboxes1[:, 0]
            bboxes1[:, 3] = bboxes1[:, 3] + bboxes1[:, 1]
            
        if len(bboxes2)>0:
            boxes_list.append(bboxes2)
            scores_list.append(scores2)
            labels_list.append(bbclasses2)
            bboxes2[:, 2] = bboxes2[:, 2] + bboxes2[:, 0]
            bboxes2[:, 3] = bboxes2[:, 3] + bboxes2[:, 1]
            
        if len(bboxes3)>0:
            boxes_list.append(bboxes3)
            scores_list.append(scores3)
            labels_list.append(bbclasses3)
            bboxes3[:, 2] = bboxes3[:, 2] + bboxes3[:, 0]
            bboxes3[:, 3] = bboxes3[:, 3] + bboxes3[:, 1]
            
        if len(bboxes4)>0:
            boxes_list.append(bboxes4)
            scores_list.append(scores4)
            labels_list.append(bbclasses4)
            bboxes4[:, 2] = bboxes4[:, 2] + bboxes4[:, 0]
            bboxes4[:, 3] = bboxes4[:, 3] + bboxes4[:, 1]

        if len(bboxes5)>0:
            boxes_list.append(bboxes5)
            scores_list.append(scores5)
            labels_list.append(bbclasses5)
            bboxes5[:, 2] = bboxes5[:, 2] + bboxes5[:, 0]
            bboxes5[:, 3] = bboxes5[:, 3] + bboxes5[:, 1]

        if len(bboxes6)>0:
            boxes_list.append(bboxes6)
            scores_list.append(scores6)
            labels_list.append(bbclasses6)
            bboxes6[:, 2] = bboxes6[:, 2] + bboxes6[:, 0]
            bboxes6[:, 3] = bboxes6[:, 3] + bboxes6[:, 1]
        
        if len(bboxes7)>0:
            boxes_list.append(bboxes7)
            scores_list.append(scores7)
            labels_list.append(bbclasses7)
            bboxes7[:, 2] = bboxes7[:, 2] + bboxes7[:, 0]
            bboxes7[:, 3] = bboxes7[:, 3] + bboxes7[:, 1]
        
        if len(bboxes8)>0:
            boxes_list.append(bboxes8)
            scores_list.append(scores8)
            labels_list.append(bbclasses8)
            bboxes8[:, 2] = bboxes8[:, 2] + bboxes8[:, 0]
            bboxes8[:, 3] = bboxes8[:, 3] + bboxes8[:, 1]
        
        if (len(bboxes1) + len(bboxes2) + len(bboxes3) + len(bboxes4) + len(bboxes5) + len(bboxes6) + len(bboxes7) + len(bboxes8)) >0:
            if Method =="NMS":
                boxes, scores, labels  = run_nms(boxes_list, scores_list, labels_list,IMG_SIZE_ENSEMBLE, iou_thr=iou_thr,weights=weights)
            elif Method =="Soft-NMS":
                boxes, scores, labels  = run_soft_nms(boxes_list, scores_list, labels_list,IMG_SIZE_ENSEMBLE, iou_thr,skip_box_thr,sigma,weights=weights)
            elif Method =="NMW":
                boxes, scores, labels  = run_nmw(boxes_list, scores_list, labels_list,IMG_SIZE_ENSEMBLE, iou_thr, skip_box_thr,weights=weights)
            elif Method =="WBF":
                boxes, scores, labels  = run_wbf(boxes_list, scores_list,labels_list,IMG_SIZE_ENSEMBLE, iou_thr, skip_box_thr,weights=weights, conf_type=WBF_conf_type)
            elif Method == "WBF_Exp":
                boxes, scores, labels  = run_wbf_exp(boxes_list, scores_list,labels_list,IMG_SIZE_ENSEMBLE, iou_thr, skip_box_thr,weights=weights, conf_type=WBF_conf_type)
            else:
                Method ="Only First Model(No Ensembling)"
                boxes, scores, labels  = bboxes1,scores1,bbclasses1 
        else:
            boxes=[]
            print("--> No object found !!")
        
        
        if len(boxes)>0 and len(boxes) == len(scores) and len(boxes) == len(labels):
            for idxx in range(len(boxes)):
                image_name.append(file_img)
                x_min.append(boxes[idxx][0]) 
                y_min.append(boxes[idxx][1])
                x_max.append(boxes[idxx][2])
                y_max.append(boxes[idxx][3])
                
                confidence.append(float(scores[idxx]))
                class_id.append(int(labels[idxx]))
        
        else:
            print("----> ", file_img)
            print(boxes)
            print(scores)
            print(labels)
    

    submission = pd.DataFrame({"image_name": image_name,
                            "class_id": class_id, 
                            "confidence_score": confidence, 
                            "x_min": x_min, 
                            "y_min": y_min, 
                            "x_max": x_max, 
                            "y_max": y_max})

    submission.to_csv(InferConfigs['CSV_Pill_Yolo_Result'], index=False)
    path_print = os.path.abspath(InferConfigs['CSV_Pill_Yolo_Result'])
    print(f'Result saved in --> {path_print}')
    print(submission.head(5))