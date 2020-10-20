

# CV 训练/测试/部署分类任务

|      支持的操作       |   具体     |    样例   |  
| :-----------------: | :---------:| :---------:|
|  模型方面  |   (efficientnet/resnest/seresnext等)       |  [1](./qdner/models/)  |
|  数据增强  |   (旋转/镜像/对比度等、mixup/cutmix)         |  [2](./qdner/dataaug/) | 
|  损失函数  |   (交叉熵/focal_loss等)                     |  [3](./qdner/loss/)    | 
|  模型部署  |   (flask/grpc/BentoML等)                   |  [4](./serving/)       | 
|  onnx/trt |   ()                                      |  [5](./tools/)         | 



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
serving
```









#

#

#

#### ref
```
（1）https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution
（2）https://github.com/BADBADBADBOY/pytorchOCR
```








