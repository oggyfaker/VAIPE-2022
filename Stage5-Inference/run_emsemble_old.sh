
git clone https://github.com/luozhouyang/python-string-similarity.git
git clone https://github.com/ultralytics/yolov5.git
git clone https://github.com/ZFTurbo/Weighted-Boxes-Fusion.git

cd ../Stage1/

python 0_Preprocess.py
python 1_Add_Names.py

cd ../Stage5-Inference/

python Ensemble.py --Configs_Inference ./Weights-FruitAI/FruitAI-Configs.yaml
python Rescore.py  --Configs_Recore ./Weights-FruitAI/FruitAI-Configs.yaml