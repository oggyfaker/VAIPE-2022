git clone https://github.com/ultralytics/yolov5 
cd yolov5/
pip install -r "requirements.txt"

# =================== Config Training With A6000 GPU (It Takes 2.5 days for complete training) ==========================
IMG_SIZE=1800
BATCH_SIZE=6
EPOCHS=70 # Best epoch is 24
WEIGHTS=yolov5x6.pt

SAVING_PATH=/root/VAIPE/pretrained/
SAVING_NAME=Yolo5x6_1800_All

YAML_DIR=/root/VAIPE/data/VAIPE-YOLOV5x6-Fold-All/VAIPE-YOLOV5x6.yaml
HYP_DIR=/root/VAIPE/data/VAIPE-YOLOV5x6-Fold-All/Hyps-Low-Default-YOLOV5x6.yaml


WANDB_MODE="dryrun" python train.py --img $IMG_SIZE \
                                    --batch $BATCH_SIZE \
                                    --epochs $EPOCHS \
                                    --weights $WEIGHTS \
                                    --data $YAML_DIR \
                                    --hyp $HYP_DIR \
                                    --project $SAVING_PATH \
                                    --name $SAVING_NAME


