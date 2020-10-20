


# CV 训练/测试/部署分类任务

|      ***       |   具体     |    样例   |  
| :-----------------: | :---------:| :---------:|
|  模型方面  |   (efficientnet/resnest/seresnext等)       |  [1](./qdnet/conf/constant.py)  |
|  数据增强  |   (旋转/镜像/对比度等、mixup/cutmix)         |  [2](./qdnet/dataaug/) | 
|  损失函数  |   (交叉熵/focal_loss等)                     |  [3](./qdnet/loss/)    | 
|  模型部署  |   (flask/grpc/BentoML等)                   |  [4](./serving/)       | 
|  onnx/trt |   ()                                      |  [5](./tools/)         | 


## 支持的全部模型：

> RESNEST_LIST = ["resnest50", "resnest101", "resnest200", "resnest269"]

> SERESNEXT_LIST = ['seresnext101']

> GEFFNET_LIST = ['GenEfficientNet', 'mnasnet_050', 'mnasnet_075', 'mnasnet_100', 'mnasnet_b1', 'mnasnet_140', 'semnasnet_050', 'semnasnet_075', 'semnasnet_100', 'mnasnet_a1', 'semnasnet_140', 'mnasnet_small','mobilenetv2_100', 'mobilenetv2_140', 'mobilenetv2_110d', 'mobilenetv2_120d', 'fbnetc_100', 'spnasnet_100', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',  'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_b8', 'efficientnet_l2', 'efficientnet_es', 'efficientnet_em', 'efficientnet_el', 'efficientnet_cc_b0_4e', 'efficientnet_cc_b0_8e', 'efficientnet_cc_b1_8e', 'efficientnet_lite0', 'efficientnet_lite1', 'efficientnet_lite2', 'efficientnet_lite3', 'efficientnet_lite4', 'tf_efficientnet_b0', 'tf_efficientnet_b1', 'tf_efficientnet_b2', 'tf_efficientnet_b3', 'tf_efficientnet_b4', 'tf_efficientnet_b5', 'tf_efficientnet_b6', 'tf_efficientnet_b7', 'tf_efficientnet_b8', 'tf_efficientnet_b0_ap', 'tf_efficientnet_b1_ap', 'tf_efficientnet_b2_ap', 'tf_efficientnet_b3_ap', 'tf_efficientnet_b4_ap', 'tf_efficientnet_b5_ap', 'tf_efficientnet_b6_ap', 'tf_efficientnet_b7_ap', 'tf_efficientnet_b8_ap', 'tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2_ns', 'tf_efficientnet_b3_ns', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns', 'tf_efficientnet_b6_ns', 'tf_efficientnet_b7_ns', 'tf_efficientnet_l2_ns', 'tf_efficientnet_l2_ns_475', 'tf_efficientnet_es', 'tf_efficientnet_em', 'tf_efficientnet_el', 'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e', 'tf_efficientnet_cc_b1_8e', 'tf_efficientnet_lite0', 'tf_efficientnet_lite1', 'tf_efficientnet_lite2', 'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 'mixnet_s', 'mixnet_m', 'mixnet_l', 'mixnet_xl', 'tf_mixnet_s', 'tf_mixnet_m', 'tf_mixnet_l']

#

## 训练/测试/部署流程：
0、转为训练需要的数据格式
```
python tools/data_preprocess.py --data_dir "./data/data.csv" --n_splits 5 --output_dir "./data/train.csv" --random_state 2020
```

1、修改配置文件，选择需要的模型 以及 模型参数：vim conf/test.yaml
```
cp conf/test.yaml conf/effb3_ns.yaml
vim conf/effb3_ns.yaml
```

2、训练模型: （根据需求选取合适的模型） 
```
python train.py --config_path "conf/effb3_ns.yaml"
```

3、测试
```
python test.py --config_path "conf/effb3_ns.yaml"
```

4、infer
```
    python infer.py --config_path "conf/effb3_ns.yaml" --img_path "./data/img/0male/0(2).jpg" --fold "0"
    >>>>> [[-5.08535   4.999839]]
    pre>>>>> [1]
    python infer.py --config_path "conf/effb3_ns.yaml" --img_path "./data/img/1female/1(5).jpg" --fold "0"
    >>>>> [[ 3.4123473 -3.469294 ]]
    pre>>>>> [0]
```

5、模型转换
```
    转onnx：python tools/pytorch_to_onnx.py
    转tensorrt：python tools/onnx_to_tensorrt.py
```

6、模型部署
```
[serving](./serving/) 
```




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
```





