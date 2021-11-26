# Update
Recently, I optimize my YOLO project:

https://github.com/yjh0410/PyTorch_YOLO-Family

On the COCO-val:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>       </th><td bgcolor=white> Backbone </td><td bgcolor=white> Size </td><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano</th><td bgcolor=white> ShuffleNetv2-1.0x </td><td bgcolor=white> 512 </td><td bgcolor=white>     </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 40.0 </td><td bgcolor=white> 20.5 </td><td bgcolor=white> 7.4 </td><td bgcolor=white> 22.7 </td><td bgcolor=white> 32.3 </td><td bgcolor=white> 1.65 </td><td bgcolor=white> 1.86M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny</th><td bgcolor=white> CSPDarkNet-Tiny </td><td bgcolor=white> 512 </td><td bgcolor=white>     </td><td bgcolor=white> 26.6 </td><td bgcolor=white> 46.1 </td><td bgcolor=white> 26.9 </td><td bgcolor=white> 13.5 </td><td bgcolor=white> 30.0 </td><td bgcolor=white> 35.0 </td><td bgcolor=white> 5.52 </td><td bgcolor=white> 7.66M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1</th><td bgcolor=white> ResNet50 </td><td bgcolor=white> 640 </td><td bgcolor=white>     </td><td bgcolor=white> 35.2 </td><td bgcolor=white> 54.7 </td><td bgcolor=white> 37.1 </td><td bgcolor=white>  14.3 </td><td bgcolor=white>  39.5 </td><td bgcolor=white>  53.4 </td><td bgcolor=white>  41.96   </td><td bgcolor=white> 44.54M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2</th><td bgcolor=white> ResNet50 </td><td bgcolor=white> 640 </td><td bgcolor=white>     </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 56.6 </td><td bgcolor=white> 37.7 </td><td bgcolor=white>  15.1 </td><td bgcolor=white>  41.1 </td><td bgcolor=white>  54.0 </td><td bgcolor=white>  42.10   </td><td bgcolor=white> 44.89M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3</th><td bgcolor=white> DarkNet53 </td><td bgcolor=white> 640 </td><td bgcolor=white>     </td><td bgcolor=white> 38.7 </td><td bgcolor=white> 60.2 </td><td bgcolor=white> 40.7 </td><td bgcolor=white>  21.3 </td><td bgcolor=white> 41.7 </td><td bgcolor=white> 51.7  </td><td bgcolor=white>  76.41   </td><td bgcolor=white> 57.25M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4</th><td bgcolor=white> CSPDarkNet53 </td><td bgcolor=white> 640 </td><td bgcolor=white>     </td><td bgcolor=white> 40.5 </td><td bgcolor=white> 60.4 </td><td bgcolor=white> 43.5 </td><td bgcolor=white> 24.2 </td><td bgcolor=white> 44.8 </td><td bgcolor=white> 52.0 </td><td bgcolor=white>  60.55   </td><td bgcolor=white> 52.00M </td></tr>

</table></tbody>

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

YOLO-Nano-1.0x:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>  <td bgcolor=white> size </td><td bgcolor=white> mAP </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 320 </td><td bgcolor=white> 65.0 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 416 </td><td bgcolor=white> 69.1 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 608 </td><td bgcolor=white> 70.8 </td></tr>
</table></tbody>

COCO:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> COCO eval</th><td bgcolor=white> 320 </td><td bgcolor=white> 17.2 </td><td bgcolor=white> 33.1 </td><td bgcolor=white> 16.2 </td><td bgcolor=white> 2.6 </td><td bgcolor=white> 16.0 </td><td bgcolor=white> 31.7 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> COCO eval</th><td bgcolor=white> 416 </td><td bgcolor=white> 19.6 </td><td bgcolor=white> 36.9 </td><td bgcolor=white> 18.6 </td><td bgcolor=white> 4.6 </td><td bgcolor=white> 19.1 </td><td bgcolor=white> 33.3 </td></tr>


<tr><th align="left" bgcolor=#f8f8f8> COCO eval</th><td bgcolor=white> 608 </td><td bgcolor=white> 20.6 </td><td bgcolor=white> 38.6 </td><td bgcolor=white> 19.5 </td><td bgcolor=white> 7.0 </td><td bgcolor=white> 22.5 </td><td bgcolor=white> 30.7 </td></tr>
</table></tbody>

YOLO-Nano-0.5x:

hold on ...


## Visualization
On COCO-val

The overview of my YOLO-Nano
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000002.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000003.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000011.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000014.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000019.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000023.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000030.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000045.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000051.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000073.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000076.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000078.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000081.jpg)
![Image](https://github.com/yjh0410/YOLO-Nano/blob/main/img_files/coco-val/000088.jpg)


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
