

Install CUDA 11.1:  https://developer.nvidia.com/cuda-11.1.0-download-archive
OR
```
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
sudo sh cuda_11.1.0_455.23.05_linux.run
```

export environment : vim ~/.bashrc
```
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda
```


Install CUDNN :  
```
wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-linux-x64-v8.2.1.32.tgz?CglCMsW0l-97IQHomlsL6lsBZ0fRvOCNZYrKZfyQWcmWjUYdtpyhc7SN5_mitLNz8v8drlztiAHqnX_D6Mw5XzanXyobErgAQo7jAY_8sbCALpKccMW26hhkBfVwb4ficnR26cTQF7qhW2QsfIIbjFooFNUWnGz7Z2_4_FaojFXfx-rhRU8msFxAr_Piv6BhzNNWOqie2gC5_eT_6s
mv cudnn-11.3-linux-x64-v8.2.1.32.tgz?CglCMsW0l-97IQHomlsL6lsBZ0fRvOCNZYrKZfyQWc.........  cudnn-11.3-linux-x64-v8.2.1.32.tgz
tar zxvf cudnn-11.3-linux-x64-v8.2.1.32.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```


Install TENSORRT(TensorRT-7.2.3.4): https://developer.nvidia.com/nvidia-tensorrt-7x-download
```
export LD_LIBRARY_PATH=/mnt/sdc/machinelp/TensorRT-7.2.3.4/lib:${LD_LIBRARY_PATH}
 
tar zxvf TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz
cd TensorRT-7.2.3.4
cd python
pip install tensorrt-7.2.3.4-cp38-none-linux_x86_64.whl
```