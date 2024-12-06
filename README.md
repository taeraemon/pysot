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

- Test tracker
```
- ~.json 다운
https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI
위에서 다 받기
일단 VOT2018.json으로 해볼거임
그거를
testing_dataset/VOT2018/VOT2018.json에 배치.

- 트래커 동작
cd experiments/siamrpn_r50_l234_dwxcorr
python3 -u ../../tools/test.py \
	--snapshot model.pth \
	--dataset VOT2018 \
	--config config.yaml

(env) tykim@tySM:~/Documents/Github-taeraemon/pysot/experiments/siamrpn_r50_l234_dwxcorr$ python3 -u ../../tools/test.py \
        --snapshot model.pth \
        --dataset VOT2018 \
        --config config.yaml
loading VOT2018:   0%|                                                | 0/60 [00:00<?, ?it/s, ants1][ WARN:0@1.390] global loadsave.cpp:241 findDecoder imread_('/home/tykim/Documents/Github-taeraemon/pysot/experiments/siamrpn_r50_l234_dwxcorr/ants1/color/00000001.jpg'): can't open/read file: check file path/integrity
loading VOT2018:   0%|                                                | 0/60 [00:00<?, ?it/s, ants1]
Traceback (most recent call last):
  File "../../tools/test.py", line 226, in <module>
    main()
  File "../../tools/test.py", line 58, in main
    load_img=False)
  File "/home/tykim/Documents/Github-taeraemon/pysot/toolkit/datasets/__init__.py", line 32, in create_dataset
    dataset = VOTDataset(**kwargs)
  File "/home/tykim/Documents/Github-taeraemon/pysot/toolkit/datasets/vot.py", line 119, in __init__
    load_img=load_img)
  File "/home/tykim/Documents/Github-taeraemon/pysot/toolkit/datasets/vot.py", line 31, in __init__
    init_rect, img_names, gt_rect, None, load_img)
  File "/home/tykim/Documents/Github-taeraemon/pysot/toolkit/datasets/video.py", line 27, in __init__
    assert img is not None, self.img_names[0]
AssertionError: /home/tykim/Documents/Github-taeraemon/pysot/experiments/siamrpn_r50_l234_dwxcorr/ants1/color/00000001.jpg

이런 에러 뜸

VOT2018 데이터 가져와야 하는것임.
개복잡함... 나중에 정리해서 따로 언급하겠음.

여튼 데이터셋 넣고 돌리면

(env) tykim@tySM:~/Documents/Github-taeraemon/pysot/experiments/siamrpn_r50_l234_dwxcorr$ python3 -u ../../tools/test.py    --snapshot model.pth    --dataset VOT2018       --config config.yaml
loading VOT2018: 100%|█████████████████████████████████| 60/60 [00:00<00:00, 185.42it/s, zebrafish1]
(  1) Video: ants1        Time:  2.2s Speed: 145.0fps Lost: 0
(  2) Video: ants3        Time:  3.4s Speed: 168.3fps Lost: 2
(  3) Video: bag          Time:  1.2s Speed: 161.4fps Lost: 0
(  4) Video: ball1        Time:  0.6s Speed: 163.3fps Lost: 0
...
( 60) Video: zebrafish1   Time:  2.5s Speed: 160.1fps Lost: 0
model total lost: 54

이런식으로 결과 나옴.
```
&nbsp;

- Eval tracker



### Environment Result
```
TBU
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
