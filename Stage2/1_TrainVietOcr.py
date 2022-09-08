import matplotlib.pyplot as plt
from PIL import Image
import os 
from os.path import join
import yaml

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

if __name__ == "__main__":
    vietocr_configs  = yaml.load(open("../configs/vietocr.yaml"), Loader=yaml.FullLoader)
    config           = Cfg.load_config_from_name("vgg_transformer")
    config['device'] = 'cuda:0'

    
    vietocr_dataset = vietocr_configs['Root-VietOCR']
    vietocr_saving  = vietocr_configs["VietOCR-Saving-Path"]
    os.makedirs(vietocr_saving, exist_ok = True)
    
    
    dataset_params = {
                'name':'hw',
                'data_root':vietocr_dataset,
                'train_annotation':'train.txt',
                'valid_annotation':'test.txt'
            }


    params = {'print_every':200,
            'valid_every':15*200,
            'iters':20000,
            'checkpoint': './checkpoint/transformerocr_checkpoint.pth',    
            'export': join(vietocr_saving, "transformerocr.pth"),
            'metrics': 10000
        }


    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = 'cuda:0'


    trainer = Trainer(config, pretrained=True)
    trainer.visualize_dataset()


    if not os.path.isfile(join(vietocr_saving, "transformerocr.pth")) and not os.path.isfile(join(vietocr_saving, "config.yml")):
        trainer.config.save(join(vietocr_saving, "config.yml"))
        print("---> Start training VietOCR Recognize Text <--- \n")
        trainer.train()
    else:
        print(f' ---> Found Trained Weights In: {vietocr_saving} <---')