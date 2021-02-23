# YOLO-Nano
A new version YOLO-Nano inspired by NanoDet.

In this project, you can enjoy: 
- a different version of YOLO-Nano


# Network
This is a a different of YOLO-Nano built by PyTorch:
- Backbone: ShuffleNet-v2
- Neck: a very lightweight FPN+PAN

# Train
- Batchsize: 32
- Base lr: 1e-3
- Max epoch: 120
- LRstep: 60, 90
- optimizer: SGD

The overview of my YOLO-Nano
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/model.png)

## Experiment
Environment:

- Python3.6, opencv-python, PyTorch1.1.0, CUDA10.0,cudnn7.5
- For training: Intel i9-9940k, RTX-2080ti

VOC:
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> mAP </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 320 </td><td bgcolor=white> - </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 416 </td><td bgcolor=white> 65.02 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 608 </td><td bgcolor=white> - </td></tr>
</table></tbody>

COCO:
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </tr>
<tr><th align="left" bgcolor=#f8f8f8> COCO val</th><td bgcolor=white> 320 </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> COCO val</th><td bgcolor=white> 416 </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> COCO val</th><td bgcolor=white> 608 </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>
</table></tbody>

## Installation
- Pytorch-gpu 1.1.0/1.2.0/1.3.0
- Tensorboard 1.14.
- opencv-python, python3.6/3.7

## Dataset

### VOC Dataset
I copy the download files from the following excellent project:
https://github.com/amdegroot/ssd.pytorch

I have uploaded the VOC2007 and VOC2012 to BaiDuYunDisk, so for researchers in China, you can download them from BaiDuYunDisk:

Link：https://pan.baidu.com/s/1tYPGCYGyC0wjpC97H-zzMQ 

Password：4la9

You will get a ```VOCdevkit.zip```, then what you need to do is just to unzip it and put it into ```data/```. After that, the whole path to VOC dataset is ```data/VOCdevkit/VOC2007``` and ```data/VOCdevkit/VOC2012```.

#### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

#### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

### MSCOCO Dataset
I copy the download files from the following excellent project:
https://github.com/DeNA/PyTorch_YOLOv3

#### Download MSCOCO 2017 dataset
Just run ```sh data/scripts/COCO2017.sh```. You will get COCO train2017, val2017, test2017.


## Train
### VOC
```Shell
python train.py -d voc --cuda -v [select a model] -ms
```

You can run ```python train.py -h``` to check all optional argument.

### COCO
```Shell
python train.py -d coco --cuda -v [select a model] -ms
```


## Test
### VOC
```Shell
python test.py -d voc --cuda -v [select a model] --trained_model [ Please input the path to model dir. ]
```

### COCO
```Shell
python test.py -d coco-val --cuda -v [select a model] --trained_model [ Please input the path to model dir. ]
```


## Evaluation
### VOC
```Shell
python eval.py -d voc --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```

### COCO
To run on COCO_val:
```Shell
python eval.py -d coco-val --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```

To run on COCO_test-dev(You must be sure that you have downloaded test2017):
```Shell
python eval.py -d coco-test --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```
You will get a .json file which can be evaluated on COCO test server.
