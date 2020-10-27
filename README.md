

# CV Easy-to-use/Easy-to-deploy/Easy-to-develop

<img src="https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif" width = "300" height = "200" alt="图片名称" align=center> <img src="https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif" width = "300" height = "200" alt="图片名称" align=center>


|      ***       |        |    example   |  
| :-----------------: | :---------:| :---------:|
|  models  |   (efficientnet/resnest/seresnext等)       |  [1](./qdnet/conf/constant.py)  |
|  metric  |   (Swish/ArcMarginProduct_subcenter/ArcFaceLossAdaptiveMargin/...)       |  [2](./qdnet/models/metric_strategy.py)  |
|  data aug  |   (rotate/flip/...、mixup/cutmix)         |  [3](./qdnet/dataaug/) | 
|  loss  |   (ce_loss/ce_smothing_loss/focal_loss/bce_loss等)                     |  [4](./qdnet/loss/)    | 
|  deploy  |   (flask/grpc/BentoML等)                   |  [5](./serving/)       | 
|  onnx/trt |   ()                                      |  [6](./tools/)         | 


## models：

> RESNEST_LIST = ["resnest50", "resnest101", "resnest200", "resnest269"]

> SERESNEXT_LIST = ['seresnext101']

> GEFFNET_LIST = ['GenEfficientNet', 'mnasnet_050', 'mnasnet_075', 'mnasnet_100', 'mnasnet_b1', 'mnasnet_140', 'semnasnet_050', 'semnasnet_075', 'semnasnet_100', 'mnasnet_a1', 'semnasnet_140', 'mnasnet_small','mobilenetv2_100', 'mobilenetv2_140', 'mobilenetv2_110d', 'mobilenetv2_120d', 'fbnetc_100', 'spnasnet_100', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',  'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_b8', 'efficientnet_l2', 'efficientnet_es', 'efficientnet_em', 'efficientnet_el', 'efficientnet_cc_b0_4e', 'efficientnet_cc_b0_8e', 'efficientnet_cc_b1_8e', 'efficientnet_lite0', 'efficientnet_lite1', 'efficientnet_lite2', 'efficientnet_lite3', 'efficientnet_lite4', 'tf_efficientnet_b0', 'tf_efficientnet_b1', 'tf_efficientnet_b2', 'tf_efficientnet_b3', 'tf_efficientnet_b4', 'tf_efficientnet_b5', 'tf_efficientnet_b6', 'tf_efficientnet_b7', 'tf_efficientnet_b8', 'tf_efficientnet_b0_ap', 'tf_efficientnet_b1_ap', 'tf_efficientnet_b2_ap', 'tf_efficientnet_b3_ap', 'tf_efficientnet_b4_ap', 'tf_efficientnet_b5_ap', 'tf_efficientnet_b6_ap', 'tf_efficientnet_b7_ap', 'tf_efficientnet_b8_ap', 'tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2_ns', 'tf_efficientnet_b3_ns', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns', 'tf_efficientnet_b6_ns', 'tf_efficientnet_b7_ns', 'tf_efficientnet_l2_ns', 'tf_efficientnet_l2_ns_475', 'tf_efficientnet_es', 'tf_efficientnet_em', 'tf_efficientnet_el', 'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e', 'tf_efficientnet_cc_b1_8e', 'tf_efficientnet_lite0', 'tf_efficientnet_lite1', 'tf_efficientnet_lite2', 'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 'mixnet_s', 'mixnet_m', 'mixnet_l', 'mixnet_xl', 'tf_mixnet_s', 'tf_mixnet_m', 'tf_mixnet_l']

#

## train/test/deploy
0、Data format transform 
```
git clone https://github.com/MachineLP/PyTorch_image_classifier
pip install -r requirements.txt
cd PyTorch_image_classifier
python tools/data_preprocess.py --data_dir "./data/data.csv" --n_splits 5 --output_dir "./data/train.csv" --random_state 2020
```

1、Modify configuration file
```
cp conf/test.yaml conf/effb3_ns.yaml
vim conf/effb3_ns.yaml
```

2、train model: 
```
python train.py --config_path "conf/effb3_ns.yaml"
```

3、test
```
python test.py --config_path "conf/effb3_ns.yaml" --n_splits 5
```

4、infer
```
    python infer.py --config_path "conf/effb3_ns.yaml" --img_path "./data/img/0male/0(2).jpg" --fold "0"
    pre>>>>> [1]
    python infer.py --config_path "conf/effb3_ns.yaml" --img_path "./data/img/1female/1(5).jpg" --fold "1"
    pre>>>>> [0]
```

5、model transform
```
    onnx：python tools/pytorch_to_onnx.py --config_path "conf/effb3_ns.yaml" --img_path "./data/img/0male/0(2).jpg" --batch_size 4 --fold 0 --save_path "lp.onnx"
    tensorrt：python tools/onnx_to_tensorrt.py
```

6、model deploy
[serving](./serving/) 





#

#

#

#

#

#

#

#### ref
```
（1）https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution
（2）https://github.com/BADBADBADBOY/pytorchOCR
（3）https://github.com/MachineLP/QDServing
（4）https://github.com/bentoml/BentoML
（5）mixup-cutmix:https://blog.csdn.net/u014365862/article/details/104216086
（7）focalloss:https://blog.csdn.net/u014365862/article/details/104216192
（8）https://blog.csdn.net/u014365862/article/details/106728375 / https://blog.csdn.net/u014365862/article/details/106728402 
```





