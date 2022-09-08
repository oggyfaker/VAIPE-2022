git clone https://github.com/ultralytics/yolov5 
cd yolov5/
pip install -r "requirements.txt"


#=================== Config Training With A6000 GPU (It Takes 6 days for complete training) ==========================
IMG_SIZE=1280
BATCH_SIZE=12
EPOCHS=80
WEIGHTS=yolov5x6.pt
SAVING_PATH=/root/VAIPE/pretrained/


#======================================================
SAVING_NAME=Yolo5x6_1280_Fold0
YAML_DIR=/root/VAIPE/data/VAIPE-YOLOV5x6-Fold0/VAIPE-YOLOV5x6.yaml
HYP_DIR=/root/VAIPE/data/VAIPE-YOLOV5x6-Fold0/Hyps-Low-Default-YOLOV5x6.yaml


WANDB_MODE="dryrun" python train.py --img $IMG_SIZE \
                                    --batch $BATCH_SIZE \
                                    --epochs $EPOCHS \
                                    --weights $WEIGHTS \
                                    --data $YAML_DIR \
                                    --hyp $HYP_DIR \
                                    --project $SAVING_PATH \
                                    --name $SAVING_NAME


#======================================================
SAVING_NAME=Yolo5x6_1280_Fold1
YAML_DIR=/root/VAIPE/data/VAIPE-YOLOV5x6-Fold1/VAIPE-YOLOV5x6.yaml
HYP_DIR=/root/VAIPE/data/VAIPE-YOLOV5x6-Fold1/Hyps-Low-Default-YOLOV5x6.yaml

WANDB_MODE="dryrun" python train.py --img $IMG_SIZE \
                                    --batch $BATCH_SIZE \
                                    --epochs $EPOCHS \
                                    --weights $WEIGHTS \
                                    --data $YAML_DIR \
                                    --hyp $HYP_DIR \
                                    --project $SAVING_PATH \
                                    --name $SAVING_NAME


#======================================================
SAVING_NAME=Yolo5x6_1280_Fold2
YAML_DIR=/root/VAIPE/data/VAIPE-YOLOV5x6-Fold2/VAIPE-YOLOV5x6.yaml
HYP_DIR=/root/VAIPE/data/VAIPE-YOLOV5x6-Fold2/Hyps-Low-Default-YOLOV5x6.yaml


WANDB_MODE="dryrun" python train.py --img $IMG_SIZE \
                                    --batch $BATCH_SIZE \
                                    --epochs $EPOCHS \
                                    --weights $WEIGHTS \
                                    --data $YAML_DIR \
                                    --hyp $HYP_DIR \
                                    --project $SAVING_PATH \
                                    --name $SAVING_NAME


#======================================================
SAVING_NAME=Yolo5x6_1280_Fold3
YAML_DIR=/root/VAIPE/data/VAIPE-YOLOV5x6-Fold3/VAIPE-YOLOV5x6.yaml
HYP_DIR=/root/VAIPE/data/VAIPE-YOLOV5x6-Fold3/Hyps-Low-Default-YOLOV5x6.yaml


WANDB_MODE="dryrun" python train.py --img $IMG_SIZE \
                                    --batch $BATCH_SIZE \
                                    --epochs $EPOCHS \
                                    --weights $WEIGHTS \
                                    --data $YAML_DIR \
                                    --hyp $HYP_DIR \
                                    --project $SAVING_PATH \
                                    --name $SAVING_NAME


#======================================================
SAVING_NAME=Yolo5x6_1280_Fold4
YAML_DIR=/root/VAIPE/data/VAIPE-YOLOV5x6-Fold4/VAIPE-YOLOV5x6.yaml
HYP_DIR=/root/VAIPE/data/VAIPE-YOLOV5x6-Fold4/Hyps-Low-Default-YOLOV5x6.yaml

WANDB_MODE="dryrun" python train.py --img $IMG_SIZE \
                                    --batch $BATCH_SIZE \
                                    --epochs $EPOCHS \
                                    --weights $WEIGHTS \
                                    --data $YAML_DIR \
                                    --hyp $HYP_DIR \
                                    --project $SAVING_PATH \
                                    --name $SAVING_NAME

