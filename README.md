# MSCMR-Pytorch-Implementation
A Multi-path Scanning Collaborative Mamba Framework with Receptive Field Prior Assignment

# Contents
- [Preliminary🔧](##Preliminary)
- [How to start💻](##Start)
- [Results☀️](##Results)
- [Acknowledgements👍](#Acknowledgements)
- [License](#License)


## Preliminary

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

## How to start


## Results


## Acknowledgements


## License
