# Custom Readme

### Overview
```
뭔가.. readme 가 친절해보이긴 한데.. 하라는 대로 해도 잘 안된다.
내 환경 : 12900KF, 2*A6000, CUDA11.8, ubuntu22.04, pyenv+venv
저는 아나콘다를 좋아하지 않습니다.
```
&nbsp;


### Overall
- 준비
```
pyenv install 3.7.6
pyenv global 3.7.6
python3 -m venv env
source env/bin/activate

pip install --upgrade pip
pip install torch
pip install opencv-python
pip install numpy
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX

python3 setup.py build_ext --inplace

export PYTHONPATH=/path/to/pysot:$PYTHONPATH
export PYTHONPATH=/home/tykim/Documents/Github-taeraemon/pysot:$PYTHONPATH
```
&nbsp;

- Demo tracker
```
- model.pth 적용

https://github.com/taeraemon/pysot/blob/master/MODEL_ZOO.md
여기서
https://drive.google.com/drive/folders/1Q4-1563iPwV6wSf_lBHDj5CPFiGSlEPG
이거 다운받기
experiments/siamrpn_r50_l234_dwxcorr/model.pth 경로로 넣기

python3 tools/demo.py \
--config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
--snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth \
--video demo/bag.avi

이렇게 해보기.

그러면 테스트 가능
```
&nbsp;

- Dataset Setup

json Setup
```
https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI
위에서 다 받기
VOT2018, OTB100을 주로 다룸.

VOT2018.json -> testing_dataset/VOT2018/VOT2018.json
OTB100.json  -> testing_dataset/VOT100/OTB100.json
```
VOT2018 Setup
```
스크립트를 활용해야하기 때문에,
https://github.com/taeraemon/VOT_Downloader
위 링크를 참조.
```
OTB100 Setup
```
http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html/
위 서버 안되어서 찾아보다가

https://figshare.com/articles/dataset/OTB2015/24427468/1?file=42879853
https://www.kaggle.com/datasets/zly1402875051/otb2015/data
https://cv.gluon.ai/build/examples_datasets/otb2015.html
위 3개의 서버로 다운로드 가능한 것을 확인 !
안돌아가서 디버깅 하다가, 마지막 서버에서 아래의 내용 확인

https://cv.gluon.ai/_downloads/719c5c0d73fb22deacc84b4557b6fd5f/otb2015.py
    os.rename(os.path.join(args,'Jogging'),os.path.join(args,'Jogging-1'))
    os.rename(os.path.join(args,'Jogging'),os.path.join(args,'Jogging-2'))
    os.rename(os.path.join(args,'Skating2'),os.path.join(args,'Skating2-1'))
    os.rename(os.path.join(args,'Skating2'),os.path.join(args,'Skating2-2'))
    os.rename(os.path.join(args,' Human4'),os.path.join(args,'Human4-2'))
위에 나온대로,
Jogging  -> Jogging-1, Jogging-2로 복사해서 만듬
Skating2 -> Skating2-1, Skating2-2로 복사해서 만듬
Human4   -> Human4-2로 이름 변경
이렇게 하니까 다 돌아감!
```
&nbsp;


- Test tracker

Dataset ready
```
아래와 같이 데이터셋을 복사해 넣어두고 진행. (VOT2018 예시)
experiments/siamrpn_r50_l234_dwxcorr
    /ants1
    /ants3
    ...
```
VOT2018 version
```
cd experiments/siamrpn_r50_l234_dwxcorr
python3 -u ../../tools/test.py \
--snapshot model.pth \
--dataset VOT2018 \
--config config.yaml
```
OTB100 version
```
cd experiments/siamrpn_r50_l234_dwxcorr
python3 -u ../../tools/test.py \
--snapshot model.pth \
--dataset OTB100 \
--config config.yaml
```
Expected Result (VOT2018 예시)
```
loading VOT2018: 100%|█████████████████████████████████| 60/60 [00:00<00:00, 185.42it/s, zebrafish1]
(  1) Video: ants1        Time:  2.2s Speed: 145.0fps Lost: 0
(  2) Video: ants3        Time:  3.4s Speed: 168.3fps Lost: 2
...
( 60) Video: zebrafish1   Time:  2.5s Speed: 160.1fps Lost: 0
model total lost: 54
```
&nbsp;

- Eval tracker

VOT2018 version
```
experiments/siamrpn_r50_l234_dwxcorr 경로에 있다고 가정
python3 ../../tools/eval.py \
--tracker_path ./results \
--dataset VOT2018 \
--num 1 \
--tracker_prefix 'model'
```
OTB100 version 
```
experiments/siamrpn_r50_l234_dwxcorr 경로에 있다고 가정
python3 ../../tools/eval.py \
--tracker_path ./results \
--dataset OTB100 \
--num 1 \
--tracker_prefix 'model'
```
Expected Result (VOT2018 예시)
```
loading VOT2018: 100%|█████████████████████████████████| 60/60 [00:00<00:00, 183.36it/s, zebrafish1]
eval ar:  100%|███████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.47it/s]
eval eao: 100%|███████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.95it/s]
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------
|   model    |  0.607   |   0.253    |    54.0     | 0.387 |
------------------------------------------------------------
```
&nbsp;


- Train tracker

ILSVRC2015 다운로드
```
wget http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz
tar -xzvf ./ILSVRC2015_VID.tar.gz

위 대로 하면 된다고 하는데, 이전에 다운받아놓은게 있어서 위에꺼 안해봄.
training_dataset/vid 밑에 작업하려면
.gitignore에 ILSVRC2015, crop511, vid.json 미리 추가해두기 (안하면 vscode 뇌정지)
```
4개 다 말고 ILSVRC2015 만 가지고 학습하는 거로 수정
```
# __C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB')
__C.DATASET.NAMES = ('VID',)

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'training_dataset/vid/crop511'
__C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

# __C.DATASET.YOUTUBEBB = CN()
# __C.DATASET.YOUTUBEBB.ROOT = 'training_dataset/yt_bb/crop511'
# __C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
# __C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
# __C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

# __C.DATASET.COCO = CN()
# __C.DATASET.COCO.ROOT = 'training_dataset/coco/crop511'
# __C.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
# __C.DATASET.COCO.FRAME_RANGE = 1
# __C.DATASET.COCO.NUM_USE = -1

# __C.DATASET.DET = CN()
# __C.DATASET.DET.ROOT = 'training_dataset/det/crop511'
# __C.DATASET.DET.ANNO = 'training_dataset/det/train.json'
# __C.DATASET.DET.FRAME_RANGE = 1
# __C.DATASET.DET.NUM_USE = -1

pysot/core/config.py를 위와같이 ILSVRC2015 만으로 학습시키려고 수정.
__C.DATASET.NAMES = ('VID',) 이런식으로 뒤에 컴마 있어야 함 (신기방기)
```
하라는대로 하기 season1
```
cd training_dataset
ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000    ILSVRC2015/Annotations/VID/train/a
ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0001    ILSVRC2015/Annotations/VID/train/b
ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0002    ILSVRC2015/Annotations/VID/train/c
ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0003    ILSVRC2015/Annotations/VID/train/d
ln -sfb $PWD/ILSVRC2015/Annotations/VID/val                                ILSVRC2015/Annotations/VID/train/e

ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000    ILSVRC2015/Data/VID/train/a
ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0001    ILSVRC2015/Data/VID/train/b
ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0002    ILSVRC2015/Data/VID/train/c
ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0003    ILSVRC2015/Data/VID/train/d
ln -sfb $PWD/ILSVRC2015/Data/VID/val                                ILSVRC2015/Data/VID/train/e
```
하라는대로 하기 season2 (아마 20분정도 소요..?)
```
python3 parse_vid.py
(xml들을 읽어서 vid.json으로 변환)

python3 par_crop.py 511 12 (26분 걸림)
(학습에 입력될 템플릿을 크롭)

python3 gen_json.py
(vid.json을 읽어서 train.json, val.json을 생성)

여튼 여기까지 제대로 안하면
FileNotFoundError: [Errno 2] No such file or directory: '/home/tykim/Documents/Github-taeraemon/pysot/pysot/datasets/../../training_dataset/vid/train.json'
이런거 떴었음.
```
이제 진짜 학습
```
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4

python3 -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=2333 \
../../tools/train.py --cfg config.yaml

내 환경은 A6000 2개
따라서 nproc_per_node는 내 GPU 개수
GPU가 메인이지 CPU가 메인이 아니라서 OMP_NUM_THREADS는 적당히 4정도.
```
Expected Result
```
[2024-12-07 23:40:02,081-rk0-train.py#249] Epoch: [1][380/9375] lr: 0.010000
        batch_time: 0.255338 (0.255625) data_time: 0.000201 (0.000221)
        cls_loss: 0.217716 (0.221119)   loc_loss: 0.300861 (0.308214)
        total_loss: 0.578749 (0.590975)
[2024-12-07 23:40:02,081-rk0-log_helper.py#105] Progress: 380 / 187500 [0%], Speed: 0.256 s/iter, ETA 0:13:17 (D:H:M)

TBU
```
&nbsp;


### Environment Result
```
(env) tykim@tySM:~/Documents/Github-taeraemon/pysot/experiments/siamrpn_r50_l234_dwxcorr$ pip list
Package                  Version
------------------------ -----------
colorama                 0.4.6
cycler                   0.11.0
Cython                   3.0.11
fonttools                4.38.0
kiwisolver               1.4.5
matplotlib               3.5.3
numpy                    1.21.6
nvidia-cublas-cu11       11.10.3.66
nvidia-cuda-nvrtc-cu11   11.7.99
nvidia-cuda-runtime-cu11 11.7.99
nvidia-cudnn-cu11        8.5.0.96
opencv-python            4.10.0.84
packaging                24.0
Pillow                   9.5.0
pip                      24.0
protobuf                 4.24.4
pyparsing                3.1.4
PyQt5-Qt5                5.15.15
PyQt5-sip                12.13.0
python-dateutil          2.9.0.post0
PyYAML                   6.0.1
setuptools               41.2.0
shapely                  2.0.6
six                      1.17.0
tensorboardX             2.6.2.2
torch                    1.13.1
tqdm                     4.67.1
typing_extensions        4.7.1
wheel                    0.42.0
yacs                     0.1.8
```
&nbsp;


### Trouble shooting

1. 빌드 문제
```
빌드 하라는 대로
python setup.py build_ext --inplace
이거 하다가 아래의 에러를 만날 수 있음.
(env) tykim@tySM:~/Documents/Github-taeraemon/pysot$ python setup.py build_ext --inplace
Compiling toolkit/utils/region.pyx because it changed.
[1/1] Cythonizing toolkit/utils/region.pyx
/home/tykim/Documents/Github-taeraemon/pysot/env/lib/python3.7/site-packages/Cython/Compiler/Main.py:381: FutureWarning: Cython directive 'language_level' not set, using '3str' for now (Py3). This has changed from earlier releases! File: /home/tykim/Documents/Github-taeraemon/pysot/toolkit/utils/region.pyx
  tree = Parsing.p_module(s, pxd, full_module_name)

Error compiling Cython file:
------------------------------------------------------------
...

from libc.stdlib cimport malloc, free
from libc.stdio cimport sprintf
from libc.string cimport strlen

cimport c_region
        ^
------------------------------------------------------------

toolkit/utils/region.pyx:11:8: 'c_region.pxd' not found

setup.py 수정하면 됨.

- before
~
setup(
    name='toolkit',
    packages=['toolkit'],
    ext_modules=cythonize(ext_modules)
)

- after
~
setup(
    name='toolkit',
    packages=['toolkit'],
    ext_modules=cythonize(
        ext_modules,
        include_path=['toolkit/utils']
    )
)
```

---
&nbsp;
&nbsp;
&nbsp;



# Origin Readme



# PySOT

**PySOT** is a software system designed by SenseTime Video Intelligence Research team. It implements state-of-the-art single object tracking algorithms, including [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html) and [SiamMask](https://arxiv.org/abs/1812.05050). It is written in Python and powered by the [PyTorch](https://pytorch.org) deep learning framework. This project also contains a Python port of toolkit for evaluating trackers.

PySOT has enabled research projects, including: [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html), [DaSiamRPN](https://arxiv.org/abs/1808.06048), [SiamRPN++](https://arxiv.org/abs/1812.11703), and [SiamMask](https://arxiv.org/abs/1812.05050).

<div align="center">
  <img src="demo/output/bag_demo.gif" width="800px" />
  <p>Example SiamFC, SiamRPN and SiamMask outputs.</p>
</div>

## Introduction

The goal of PySOT is to provide a high-quality, high-performance codebase for visual tracking *research*. It is designed to be flexible in order to support rapid implementation and evaluation of novel research. PySOT includes implementations of the following visual tracking algorithms:

- [SiamMask](https://arxiv.org/abs/1812.05050)
- [SiamRPN++](https://arxiv.org/abs/1812.11703)
- [DaSiamRPN](https://arxiv.org/abs/1808.06048)
- [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html)
- [SiamFC](https://arxiv.org/abs/1606.09549)

using the following backbone network architectures:

- [ResNet{18, 34, 50}](https://arxiv.org/abs/1512.03385)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

Additional backbone architectures may be easily implemented. For more details about these models, please see [References](#references) below.

Evaluation toolkit can support the following datasets:

:paperclip: [OTB2015](http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf) 
:paperclip: [VOT16/18/19](http://votchallenge.net) 
:paperclip: [VOT18-LT](http://votchallenge.net/vot2018/index.html) 
:paperclip: [LaSOT](https://arxiv.org/pdf/1809.07845.pdf) 
:paperclip: [UAV123](https://arxiv.org/pdf/1804.00518.pdf)

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [PySOT Model Zoo](MODEL_ZOO.md).

## Installation

Please find installation instructions for PyTorch and PySOT in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using PySOT

### Add PySOT to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
```

### Download models
Download models in [PySOT Model Zoo](MODEL_ZOO.md) and put the model.pth in the correct directory in experiments

### Webcam demo
```bash
python tools/demo.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth
    # --video demo/bag.avi # (in case you don't have webcam)
```

### Download testing datasets
Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker
```bash
cd experiments/siamrpn_r50_l234_dwxcorr
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset VOT2018 	\ # dataset name
	--config config.yaml	  # config file
```
The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker
assume still in experiments/siamrpn_r50_l234_dwxcorr_8gpu
``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
```

###  Training :wrench:
See [TRAIN.md](TRAIN.md) for detailed instruction.


### Getting Help :hammer:
If you meet problem, try searching our GitHub issues first. We intend the issues page to be a forum in which the community collectively troubleshoots problems. But please do **not** post **duplicate** issues. If you have similar issue that has been closed, you can reopen it.

- `ModuleNotFoundError: No module named 'pysot'`

:dart:Solution: Run `export PYTHONPATH=path/to/pysot` first before you run the code.

- `ImportError: cannot import name region`

:dart:Solution: Build `region` by `python setup.py build_ext —-inplace` as decribled in [INSTALL.md](INSTALL.md).


## References

- [Fast Online Object Tracking and Segmentation: A Unifying Approach](https://arxiv.org/abs/1812.05050).
  Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, Philip H.S. Torr.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

- [SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks](https://arxiv.org/abs/1812.11703).
  Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, Junjie Yan.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

- [Distractor-aware Siamese Networks for Visual Object Tracking](https://arxiv.org/abs/1808.06048).
  Zheng Zhu, Qiang Wang, Bo Li, Wu Wei, Junjie Yan, Weiming Hu.
  The European Conference on Computer Vision (ECCV), 2018.

- [High Performance Visual Tracking with Siamese Region Proposal Network](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html).
  Bo Li, Wei Wu, Zheng Zhu, Junjie Yan, Xiaolin Hu.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

- [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549).
  Luca Bertinetto, Jack Valmadre, João F. Henriques, Andrea Vedaldi, Philip H. S. Torr.
  The European Conference on Computer Vision (ECCV) Workshops, 2016.
  
## Contributors

- [Fangyi Zhang](https://github.com/StrangerZhang)
- [Qiang Wang](http://www.robots.ox.ac.uk/~qwang/)
- [Bo Li](http://bo-li.info/)
- [Zhiyuan Chen](https://zyc.ai/)
- [Jinghao Zhou](https://shallowtoil.github.io/)

## License

PySOT is released under the [Apache 2.0 license](https://github.com/STVIR/pysot/blob/master/LICENSE). 
