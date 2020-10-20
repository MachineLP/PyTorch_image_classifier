#-*- coding:utf-8 _*-
import sys
sys.path.append('./')
import yaml
import math 
import argparse
import torch.nn as nn
import torch
import cv2
import numpy as np
import onnx
import time
import onnxruntime
from PIL import Image
from qdnet.dataaug.dataaug import get_transforms
from qdnet.conf.config import load_yaml
from ptocr.utils.util_function import create_module,load_model

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path', help='config file path')
args = parser.parse_args()
config = load_yaml(args.config_path, args)


if config["enet_type"] == 'resnest101':
    ModelClass = Resnest
elif config["enet_type"] == 'seresnext101':
    ModelClass = SeResnext
elif 'efficientnet' in config["enet_type"]:  
    ModelClass = Effnet
else:
    raise NotImplementedError()

model = ModelClass(
        config["enet_type"],
        out_dim=config["out_dim"],
        pretrained=config["pretrained"] )     
model = model.to(device)


def gen_onnx(args):
    stream = open(args.config, 'r', encoding='utf-8')


    if config["eval"] == 'best':    
        model_file = os.path.join(config["model_dir"], f'best_fold{fold}.pth')
    if config["eval"] == 'final':
        model_file = os.path.join(config["model_dir"], f'final_fold{fold}.pth')


    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_file), strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_file)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)

    model.eval()

    print('load model ok.....')
    

    img = cv2.imread(args.img_path)
    img1 = Image.fromarray(img).convert('RGB')
    transforms_train, transforms_val = get_transforms(config["image_size"])   
    img1 = transforms.ToTensor()(img1)
    img1 = transforms_val(image=img1)

    s = time.time()
    with torch.no_grad():
        out = model(img1)
    print('cost time:',time.time()-s)
    if isinstance(out,dict):
        out = out['f_score']
    
    cv2.imwrite('./onnx/ori_output.jpg',out[0,0].cpu().detach().numpy()*255)

    output_onnx = args.save_path
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input"]
    # output_names = ["hm" , "wh"  , "reg"]
    output_names = ["out"]
    inputs = torch.randn(args.batch_size, 3,w,h).cuda()
    torch_out = torch.onnx._export(model, inputs, output_onnx, export_params=True, verbose=False,do_constant_folding=False,keep_initializers_as_inputs=True,
                                   input_names=input_names, output_names=output_names)


    onnx_path = args.save_path
    session = onnxruntime.InferenceSession(onnx_path)
    # session.get_modelmeta()
    # input_name = session.get_inputs()[0].name
    # output_name = session.get_outputs()[0].name

    image = img / 255.0
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # image = (image - mean) / std
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    s = time.time()
    preds = session.run(['out'], {'input': image})
    preds = preds[0]
    print(time.time()-s)
    if isinstance(preds,dict):
        preds = preds['f_score']
    cv2.imwrite('./onnx/onnx_output.jpg',preds[0,0]*255)

    print('error_distance:',np.abs((out.cpu().detach().numpy()-preds)).mean())
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--img_path', nargs='?', type=str, default=None)
    parser.add_argument('--save_path', nargs='?', type=str, default=None)
    parser.add_argument('--batch_size', nargs='?', type=int, default=None)
    parser.add_argument('--max_size', nargs='?', type=int, default=None)
    parser.add_argument('--algorithm', nargs='?', type=str, default=None)
    parser.add_argument('--add_padding', action='store_true', default=False)
    
    args = parser.parse_args()
    gen_onnx(args)