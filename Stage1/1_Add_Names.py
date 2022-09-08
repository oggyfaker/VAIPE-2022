import json 
import os 
import pandas as pd 
import yaml 
from tqdm import tqdm 


if __name__ == "__main__":
    #============= 0.Read configs =============
    configs = yaml.load(open("../configs/data.yaml"), Loader=yaml.FullLoader)

    Root_Path_MedicalBill_Label = configs['Path-Pres-Label']
    Json_Mapping_File           = configs['Json-Pres-Img']
    Train_Detection_CSV         = configs['Train-Detection-CSV']
    Train_Detection_Names_CSV   = configs['Train-Detection-Name-CSV']
    
    #============= 1.Create Mapping =============
    new_mapping_json = {}
    with open(Json_Mapping_File, 'r') as f:
        mapping_json = json.load(f)    
    
    for item in mapping_json:
        new_mapping_json[str(item['pres'])] = item['pill']

    data = pd.read_csv(Train_Detection_CSV).reset_index(drop=True)


    #============= 2.Start add name to train detection csv =============
    Path = Root_Path_MedicalBill_Label


    for idx,json_file in enumerate(tqdm(os.listdir(Path))):
        with open(os.path.join(Path, json_file), 'r') as f:
            bill = json.load(f)
        
        related_files = new_mapping_json[json_file]
        for item in bill:
            if "mapping" in item.keys():
                label_num = item["mapping"]
                text_label= " ".join(item['text'].split(" ")[1:])
                
                for file in related_files:
                    file = file.replace(".json", "")
                    data.loc[(data["image_id"]==file) & (data['class_id']==label_num), 'names'] = text_label

    #============= 3.Saving =============
    data = data.reset_index(drop=True)
    if not os.path.exists(Train_Detection_Names_CSV): 
        data.to_csv(Train_Detection_Names_CSV, index=False)
    
    print(data.head(5))
    