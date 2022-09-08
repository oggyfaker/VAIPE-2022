# Make Fold dataset 

python Make_Yolo_Pill_Dataset.py --Yaml_Data_VAIPE ../configs/data.yaml \
                                --Yaml_Data_Yolo ../configs/Yolo5x6_All.yaml

python Make_Yolo_Pill_Dataset.py --Yaml_Data_VAIPE ../configs/data.yaml \
                                --Yaml_Data_Yolo ../configs/Yolo5x6_Fold0.yaml

python Make_Yolo_Pill_Dataset.py --Yaml_Data_VAIPE ../configs/data.yaml \
                                --Yaml_Data_Yolo ../configs/Yolo5x6_Fold1.yaml

python Make_Yolo_Pill_Dataset.py --Yaml_Data_VAIPE ../configs/data.yaml \
                                --Yaml_Data_Yolo ../configs/Yolo5x6_Fold2.yaml

python Make_Yolo_Pill_Dataset.py --Yaml_Data_VAIPE ../configs/data.yaml \
                                --Yaml_Data_Yolo ../configs/Yolo5x6_Fold3.yaml

python Make_Yolo_Pill_Dataset.py --Yaml_Data_VAIPE ../configs/data.yaml \
                                --Yaml_Data_Yolo ../configs/Yolo5x6_Fold4.yaml