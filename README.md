# VAIPE-2022
VAIPE Challenge 2022

 ## 1. Directory & file structures
```
VAIPE-2022
│   .gitignore
│   README.md
│   requirements.txt
|   Dockerfile            # file setup library for Docker
│   Run_Docker_Build.sh   # train - create old submission - create new submission
│
└───configs/ - Configs dir để tạo ra các data để train model yolov5x6 - yoloocr - VietOCR
|   |    data.yaml
|   |    vietocr.yaml
|   |    yolo-ocr.yaml
|   |    Yolo5x6_All.yaml
|   |    Yolo5x6_Fold0.yaml 
|   |    Yolo5x6_Fold1.yaml
|   |    Yolo5x6_Fold2.yaml
|   |    Yolo5x6_Fold3.yaml
|   |    Yolo5x6_Fold4.yaml
|   |
└───data/ - datasets from challenge 
|   |   
│   │   public_test
|   |   |--   pill
|   |   |--   prescription
|   |   |--   pill_pres_map.json
|   |   |--   pubval_groundtruth.csv
|   |   
│   │   public_test_new
|   |   |--   pill
|   |   |--   prescription
|   |   |--   pill_pres_map.json
|   |   
│   │   public_train
|   |   |--   pill
|   |   |--   prescription
|   |   |--   pill_pres_map.json
|
│
└───Stage1/ - Create CSV dataset 
|   |   
│   │   0_Preprocess.py # Create ./data/pulic_train/train_detection.csv
|   |   1_Add_Names.py  # Create ./data/pulic_train/train_detection_with_name.csv
|   |
|   
└───Stage2/ - Train Model VietOCR
|   |   
│   │   0_CreateLabel.py   # Create ./data/VAIPE_VietOCR
|   |   1_TrainVietOcr.py  # Train Model VietOCR for Text Recognition
|   |
|   
└───Stage3/ - Train YoloOCR 
|   |   
│   │   0_Train_Yolo_OCR.sh     # Bash run Train Yolo for Text Detection  
|   |   MakeDataset_YoloOCR.py  # Make dataset For Yolo detection text 
|   |
|   
└───Stage4/ - Train Yolo5x6 Multiscale Size For Pill Detection  
|   |   
│   │   0_Make_Yolo_Pill_Fold.sh     # Bash Run Make Multi Fold Dataset For Yolo5x6 Training 
│   │   Make_Yolo_Pill_Dataset.py 
|   |   1_Yolo5x6_1280.sh    # Bash Run Train 1280x1280
│   │   2_Yolo5x6_1800.sh    # Bash Run Train 1800x1800  
│   │   3_Yolo5x6_2432.sh    # Bash Run Train 2432x2432
│   │   4_Yolo5x6_3000.sh    # Bash Run Train 3000x3000
|   |
|   
└───Stage5-Inference/ - Inference Model 
|   |   
|   |   Weights-FruitAI (Download from googledrive)
│   │   Ensemble.py  # Inference Yolov5x6 Multiscale Pill Detection   
|   |   Rescore.py   # Make dataset For Yolo detection text 
|   |   run_ensemble_old.sh  # Bash run Ensemble by FruitAI Weights  
|   |
|   
```

 ## 2. Setup Dataset & Dockers
```
  ## Setup Docker and Docker-nvida2 (Ubuntu==22.04: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  ## Git clone source &  download dataset từ challenge bỏ vào thư mục như sơ đồ phía trên miêu tả
  
  $ git clone https://github.com/oggyfaker/VAIPE-2022
  $ cd VAIPE-2022/
  

  ## Trước khi run docker build, vào Run_Docker_Build.sh thay đổi:  
  ## $NAME_DOCKER (có thể giữ nguyên để dùng ^^)
  ## $REPO_PATH (Thay đổi với absolute path dẫn đến REPO : ví dụ $DIR/VAIPE-2022)
  
  $ sh Run_Docker_Build.sh   ## Folder local sẽ được mount trong thư mục docker: /root/VAIPE/
  
  
  ## Khi vào được docker chạy lệnh sau: 
  $ source activate VAIPE2022
  $ cd /root/VAIPE 
  
  ## Sau khi hoàn thành, có thể chạy dc các lệnh dưới 
```

## Start Training Model 

 ### 3. Create CSV file  
```
  ## Tạo file ./data/pulic_train/train_detection.csv && ./data/pulic_train/train_detection_with_name.csv
  $ cd Stage1/
  $ python 0_Preprocess.py
  $ python 1_Add_Names.py 
```

### 4. Train VietOCR Text Recognition  
```
  $ cd Stage2/
  $ python 0_CreateLabel.py
  $ python 1_TrainVietOcr.py 
```

### 5. Train YoloOCR Text Detection 
```
  $ cd Stage3/
  $ sh 0_Train_Yolo_OCR.sh
```

### 6. Train Yolo Text Detection 
```
  ## Để có thể run tùy model, trong mỗi file bash run sh , chỉnh Batchsize
  ## Mọi model run đều sẽ save weights trong thư mục /VAIPE-2022/pretrained
  
  $ cd Stage4/
  $ sh 0_Make_Yolo_Pill_Fold.sh 
  $ sh 1_Yolo5x6_1280.sh
  $ sh 2_Yolo5x6_1800.sh
  $ sh 3_Yolo5x6_2432.sh 
  $ sh 4_Yolo5x6_3000.sh
```
## Inference By FruitAI Weights 

```
  ## Download, Unzip Weights-FruiAI vào trong thư mục /VAIPE/Stage5-Inference như trong 1. Directory & file structures 
  ## File csv của Pill Detection sẽ được save trong /VAIPE-2022/Stage5-Inference/results_detection_pill.csv
  ## File csv của Rescore (File csv cuối cùng) sẽ được save trong /VAIPE-2022/Stage5-Inference/results.csv
  ## results.csv là file được dùng để submit 
  
  $ cd Stage5-Inference/
  $ sh run_ensemble_old.sh
```

## Inference By Reproduce Training 
```
  ## Download, Unzip Weights-FruiAI từ google drive
  ## Vào trong thư mục /VAIPE/Stage5-Inference như trong 1.Directory & file structures
  ## Khởi tạo 1 folder với tên và foler giống Weights-FruiAI
  
  ## Vào /VAIPE-2022/pretrained copy đúng các weights từ yolo5x6 training và yoloOCR và VietOCR đã train được 
  ## Đổi tên giống các file trong Weights-FruiAI
  ## Copy file configs từ Weights-FruiAI bỏ vào thư mục
  ## Có thể bắt đầu Inference bằng Reproduce Training  
  
  $ cd Stage5-Inference/
  $ sh run_ensemble_old.sh 
```

 
