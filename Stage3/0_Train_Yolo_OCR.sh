#!/bin/bash

# Make YoloOCR Dataset 
python MakeDataset_YoloOCR.py


# Train YoloOCR 
git clone https://github.com/ultralytics/yolov5 
cd yolov5/
pip install -r "requirements.txt"


IMG_SIZE=640 
BATCH_SIZE=10
EPOCHS=40

SAVING_NAME=Yolo5x_OCR_Weights
YAML_DIR=/root/VAIPE/data/Yolov5-OCR/Yolov5-OCR.yaml
SAVING_PATH=/root/VAIPE/pretrained/

WANDB_MODE="dryrun" python train.py --img $IMG_SIZE \
                                    --batch $BATCH_SIZE \
                                    --epochs $EPOCHS \
                                    --weights yolov5x.pt \
                                    --data $YAML_DIR \
                                    --project $SAVING_PATH \
                                    --name $SAVING_NAME
