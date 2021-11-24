from __future__ import division

import os
import argparse
import time
import random
import cv2
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.voc import VOCDetection
from data.coco import COCODataset
from data import config
from data.transforms import TrainTransforms, ColorTransforms, ValTransforms

from utils.com_flops_params import FLOPs_and_Params
from utils.misc import detection_collate, ModelEMA

from evaluator.cocoapi_evaluator import COCOAPIEvaluator
from evaluator.vocapi_evaluator import VOCAPIEvaluator

import tools


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-Nano Detection')
    # Basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--img_size', default=640, type=int, 
                        help='input image size')
    parser.add_argument('--multi_scale_range', nargs='+', default=[10, 20], type=int,
                        help='lr epoch to decay')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--max_epoch', type=int, default=150,
                        help='The upper bound of warm-up')
    parser.add_argument('--lr_epoch', nargs='+', default=[90, 120], type=int,
                        help='lr epoch to decay')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')
    # Model
    parser.add_argument('-v', '--version', default='yolo_nano',
                        help='yolo_nano,.')
    # Dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    # Train trick
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema training trick')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='use mosaic augmentation.')

    return parser.parse_args()


def train():
    args = parse_args()

    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # multi-scale
    train_size = val_size = args.img_size

    cfg = config.train_cfg
    # dataset and evaluator
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    print('Loading the dataset...')

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(args, train_size, val_size, device)
    # dataloader
    dataloader = build_dataloader(args, dataset, detection_collate)
    
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    if args.dataset == 'voc':
        anchor_size = config.MULTI_ANCHOR_SIZE
    elif args.dataset == 'coco':
        anchor_size = config.MULTI_ANCHOR_SIZE_COCO

    # build model
    if args.version == 'yolo_nano':
        from models.yolo_nano import YOLONano
        backbone = '1.0x'
        net = YOLONano(device=device,
                       input_size=train_size,
                       num_classes=num_classes, 
                       trainable=True, 
                       anchor_size=anchor_size, 
                       backbone=backbone
                       )
        print('Let us train yolo_nano ......')

    else:
        print('Unknown model version !!!')
        exit()

    model = net
    model = model.to(device)

    # compute FLOPs and Params
    model.trainable = False
    model.eval()
    FLOPs_and_Params(model=model, size=train_size, device=device)
    model.trainable = True
    model.train()

    # EMA
    ema = ModelEMA(model) if args.ema else None

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)
    
    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=0.9,
                            weight_decay=5e-4
                            )

    max_epoch = args.max_epoch
    epoch_size = len(dataset) // args.batch_size
    warmup = True

    # start training loop
    t0 = time.time()

    for epoch in range(args.start_epoch, max_epoch):

        # use step lr
        if epoch in args.lr_epoch:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        for iter_i, (images, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            if epoch < args.wp_epoch and warmup:
                nw = args.wp_epoch * epoch_size
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0 and warmup:
                # warmup is over
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)        

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                train_size = random.randint(10, 19) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            targets = [label.tolist() for label in targets]
            # make train label
            targets = tools.multi_gt_creator(train_size, net.stride, targets, anchor_size=anchor_size)
            
            # to device
            images = images.to(device).float()
            targets = targets.to(device).float()

            # forward and loss
            conf_loss, cls_loss, bbox_loss, iou_loss = model(images, target=targets)

            # total loss
            total_loss = conf_loss + cls_loss + bbox_loss + iou_loss

            # check loss
            if torch.isnan(total_loss):
                continue

            # Backward and Optimize
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            # ema
            if args.ema:
                ema.update(model)

            # display
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('obj loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('cls loss', cls_loss.item(),  iter_i + epoch * epoch_size)
                    writer.add_scalar('box loss', bbox_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('iou loss',  iou_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || iou %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), 
                            cls_loss.item(), 
                            bbox_loss.item(), 
                            iou_loss.item(),
                            total_loss.item(), 
                            train_size, 
                            t1-t0),
                        flush=True)

                t0 = time.time()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0:
            model.trainable = False
            model.set_grid(val_size)
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            # convert to training mode.
            model.trainable = True
            model.set_grid(train_size)
            model.train()

            # save model
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, 
                        args.version + '_' + repr(epoch + 1) + '.pth')
                        )  


def build_dataset(args, train_size, val_size, device):
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        dataset = VOCDetection(
                        data_dir=data_dir,
                        img_size=train_size,
                        transform=TrainTransforms(train_size),
                        color_augment=ColorTransforms(train_size),
                        mosaic=args.mosaic)

        evaluator = VOCAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size))

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=train_size,
                    image_set='train2017',
                    transform=TrainTransforms(train_size),
                    color_augment=ColorTransforms(train_size),
                    mosaic=args.mosaic)

        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size)
                        )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, collate_fn=None):
    # dataloader
    dataloader = torch.utils.data.DataLoader(
                    dataset=dataset, 
                    shuffle=True,
                    batch_size=args.batch_size, 
                    collate_fn=collate_fn,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )
    return dataloader


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
