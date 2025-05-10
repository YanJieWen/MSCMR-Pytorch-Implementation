# CVMR-Pytorch-Implementation
A Multi-path Scanning Collaborative Mamba Framework with Receptive Field Prior Assignment

# Contents
- [Preliminary🔧](#Preliminary)
- [How to start💻](#Start)
- [Results☀️](#Results)
- [Acknowledgements👍](#Acknowledgements)
- [License](#License)


## Preliminary🔧

### Requirments
Causal CNN[![causalCNN](https://img.shields.io/badge/CUDA-CNN-blue)](https://github.com/Dao-AILab/causal-conv1d/releases)  
Selective State Space Model (S6)[![S6](https://img.shields.io/badge/CUDA-S6-blue)](https://github.com/state-spaces/mamba/releases)  
`
Cuda version must be aligen with python&pytorch. 
` We can get more detials from this [Web](https://github.com/state-spaces/mamba/issues/97)


### Pretrained 
The pretrianed `pt` weights can be obtained in the [Ultralytics](https://docs.ultralytics.com/zh/models/yolov8/#overview), and put them into the `pretrained` root.  
The datasets [COCO](https://cocodataset.org/) and homemade [Crash2024](https://drive.google.com/drive/folders/1BJOdywj-hgXRKt_q0TEcBGpCV4Wojmhc?usp=drive_link). Putting them into the `datasets` root.  
`COCO-like as JSON type is suggested`

### Download
```
#download our code
git clone https://github.com/YanJieWen/MSCMR-Pytorch-Implementation.git
#into the repository
cd MSCMR-Pytorch-Implementation
#change version (can be ignored)
git checkout tags/XXX
#It is recommended to run the code in edit mode
pip install -v -e .
```

## How to start💻

### Changes
Our code is built on [Ultralytics](https://github.com/ultralytics/ultralytics).
The extral changes are:  

[CoMamba block](https://github.com/YanJieWen/MSCMR-Pytorch-Implementation/tree/master/ultralytics/nn/comamba)  

[PLA assigment](https://github.com/YanJieWen/MSCMR-Pytorch-Implementation/blob/master/ultralytics/utils/tal.py)  

[KL divergence cost](https://github.com/YanJieWen/MSCMR-Pytorch-Implementation/blob/master/ultralytics/utils/metrics.py)  

[Visual tools](https://github.com/YanJieWen/MSCMR-Pytorch-Implementation/tree/master/tools)

### Overall Framework
MSCMR are built based on [cfg](https://github.com/YanJieWen/MSCMR-Pytorch-Implementation/tree/master/ultralytics/cfg/models/comambayolo).  
![image](assets/image_1.jpg)

Our key contribution is 2DVmamba:
![image](assets/image_2.jpg)


### Training
Our MSCMR is follow [YOLOv8](https://github.com/ultralytics/ultralytics), there are 2 version supported: `Nano` and `Small`.  
Taking `Small` version as example：
1) change [tal.py](https://github.com/YanJieWen/MSCMR-Pytorch-Implementation/blob/master/ultralytics/utils/tal.py) -->``line 369`` aligen the model version

2) change [datasets cfg](https://github.com/YanJieWen/MSCMR-Pytorch-Implementation/tree/master/ultralytics/cfg/datasets)

3) ``model=YOLO(./ultralytics/cfg/models/comambayolo/comamba-s.yaml)``

4) ``model.train(data='./ultralytics/cfg/datasets/crash2024.yaml',batch=16,epochs=24,device=[0],lr0=0.01,pretrained='./pretrained/yolov8s-cls.pt')``




### Inference
We train MSCMR on COCO, you can get the pretrained weight from:  
[![MSCMR-Nano-COCO](https://img.shields.io/badge/MSCMR-Nano-red)](https://drive.google.com/drive/folders/1ibZXjqyxoHPkNSKeLkd1HRNTG5tQk3SC?usp=drive_link)

[![MSCMR-Small-COCO](https://img.shields.io/badge/MSCMR-Small-red)](https://drive.google.com/drive/folders/1g9EDylrVlyOiNVCvcPetZE71mP_0HByL?usp=drive_link)  

We provide a variety of visualization services in [analysis.py](analysis.py), which is connected on [tools](https://github.com/YanJieWen/MSCMR-Pytorch-Implementation/tree/master/tools)

1) get results which are saved as json-->[save_json.py](tools/save_json.py)
2) get effective receptive field -->[get_erf.py](tools/get_erf.py)
3) most popular heatmaps based on grad-cam-->[get_heat.py](tools/get_heat.py)
4) visual based on json file-->[get_bboxes.py](tools/get_bboxes.py)
5) draw TP,FP,FN based on json file-->[get_detection.py](tools/get_detection.py)

## Results☀️
Version | AP | AP50
---- | ---- | ---- 
MSCMR-N | 39.6 | 55.6
YOLOv8-N | 36.4 | 51.3
MSCMR-S | 45.8 | 62.9
YOLOv8-S | 44.0 | 60.1


## Acknowledgements👍
  ### Papers
  [Mamba](https://arxiv.org/abs/2312.00752)  
  
  [vision Mamba](https://arxiv.org/abs/2401.09417)  
  
  [VMamaba](https://arxiv.org/abs/2401.10166)  
  
  [YOLOv10](https://arxiv.org/abs/2405.14458)  
  
  [YOLOMamba](https://arxiv.org/abs/2406.05835)  
  
  [Grad-CAM](https://link.springer.com/article/10.1007/S11263-019-01228-7)

  ### Codes
  [VMamba](https://github.com/MzeroMiko/VMamba)  
  
  [Ultralytics](https://github.com/ultralytics/ultralytics)  
  
  [YOLOMamba](https://github.com/HZAI-ZJNU/Mamba-YOLO)  
  
  [YOLOv10](https://github.com/THU-MIG/yolov10)  
  
  [mmdetection](https://github.com/open-mmlab/mmdetection)  
  
  [mmYOLO](https://github.com/open-mmlab/mmyolo)  

  [Detection tools](https://github.com/z1069614715/objectdetection_script)


## License
[MIT](LICENSE) © YanjieWen

