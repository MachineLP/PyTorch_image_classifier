#-*- coding:utf-8 _*-
import os
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  
import onnx
import cv2
import numpy as np
import time
import argparse
import math
from onnx_tensorrt.tensorrt_engine import Engine
from qdnet.dataaug.dataaug import get_transforms


parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--img_path', nargs='?', type=str, default=None)
parser.add_argument('--config_path', help='config file path')
parser.add_argument('--batch_size', nargs='?', type=int, default=None)
parser.add_argument('--fold', help='config file path', type=int)
parser.add_argument('--save_path', help='config file path', type=str)
parser.add_argument('--trt_save_path', help='config file path', type=str)
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
device = torch.device('cuda')
model = model.to(device)

def gen_trt_engine(args):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING) 
    model = onnx.load(args.save_path)
    model_str = model.SerializeToString()
    builder = trt.Builder(TRT_LOGGER)
    builder.max_batch_size = config['batch_size']
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    # if builder.platform_has_fast_int8:
    #     builder.int8_mode = True
    networks = builder.create_network(flags=1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(networks, TRT_LOGGER)
    if not parser.parse(model_str):
        raise ValueError('parse onnx model fail')
    for layer_idx in range(networks.num_layers):
        layer = networks.get_layer(layer_idx)
        if layer.precision == trt.DataType.FLOAT:
            layer.precision = trt.DataType.HALF
            print('conver {} to HALF'.format(layer.name))
        elif layer.precision == trt.DataType.INT32:
            layer.precision = trt.DataType.INT32
        else:
            layer.precision = layer.precision
    inputs = networks.get_input(0)
    inputs.dtype = trt.DataType.HALF
    # outputs = networks.get_output(0)
    # outputs.dtype = trt.DataType.HALF
    engine = builder.build_cuda_engine(networks)
    with open(args.trt_save_path, 'wb') as f:
        f.write(engine.serialize())
    return engine

class ModelTensorRT:
    def __init__(self):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        engine = gen_trt_engine(args)
        self.engine = Engine(engine)
        self.inputs_shape = self.engine.inputs[0].shape
        print('engine input shape', self.inputs_shape)

    def less_predict(self, inputs):
        print('inputs batch less than engine inputs')
        inp_batch = inputs.shape[0]
        inputs = np.vstack([inputs, np.zeros((self.inputs_shape[0] - inp_batch, *self.inputs_shape[1:]),
                                             dtype=np.float16)])
        outputs = self.engine.run([inputs])[0]
        outputs = outputs[:inp_batch, :]
        return outputs

    def forward(self, img_path):
        try:
            img = cv2.imread(img_path)  
            transforms_train, transforms_val = get_transforms(config["image_size"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = transforms_val(image=img)
            img1 = res['image'].astype(np.float32)
            img1 = img1.transpose(2, 0, 1)
            inputs = img1.cpu().numpy()
            inputs = np.array(inputs, copy=True, dtype=np.float16)
            inp_batch = inputs.shape[0]
            if inp_batch < self.inputs_shape[0]:
                outputs = self.less_predict(inputs)
            elif inp_batch == self.inputs_shape[0]:
                print('batch size equal ')
                outputs = self.engine.run([inputs])[0]
            else:
                print('inputs batch greater than engine inputs')
                outputs = []
                ixs = list(range(0, inp_batch, self.inputs_shape[0])) + [inp_batch]
                for i in ixs:
                    if i != 0:
                        inp = inputs[li:i, :]
                        if inp.shape[0] == self.inputs_shape[0]:
                            outs = self.engine.run([inp])[0]
                        else:
                            outs = self.less_predict(inp)
                        t = outs.copy()
                        outputs.append(t)
                    li = i
                outputs = np.vstack(outputs)
            outputs = torch.tensor(outputs)
            outputs = F.softmax(outputs, dim=1).cpu()
            score, class_prediction = torch.max(outputs, 1)
            return score, class_prediction
        except Exception as e:
            raise e



if __name__ == "__main__":    
    # gen_trt_engine(args)
    m_trt = ModelTensorRT()
    img_path = args.img_path
    score, class_prediction = m_trt.forward(img_path)
    print (score, class_prediction)
