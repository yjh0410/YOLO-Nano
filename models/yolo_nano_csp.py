import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv, SPP, DWBottleneckCSP, UpSample
from backbone import *
import numpy as np
import tools


class YOLONano_CSP(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.50, anchor_size=None, backbone='1.0x', diou_nms=False):
        super(YOLONano_CSP, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.nms_processor = self.diou_nms if diou_nms else self.nms
        self.bk = backbone
        self.stride = [8, 16, 32]
        self.anchor_size = torch.tensor(anchor_size).view(3, len(anchor_size) // 3, 2)
        self.num_anchors = self.anchor_size.size(1)

        self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)

        # backbone: shufflenet-v2
        if self.bk == '1.0x':
            # use shufflenetv2_1.0x as backbone
            print('Use backbone: shufflenetv2_1.0x')
            self.backbone = shufflenetv2(model_size=self.bk, pretrained=trainable)
            c3, c4, c5 = 116, 232, 464
        
        else:
            print("For YOLO-Nano, we only support <0.5x, 1.0x> as our backbone !!")
            exit(0)

        # 1x1 conv
        self.conv1x1_0 = Conv(c3, 96, k=1, leaky=False)
        self.conv1x1_1 = Conv(c4, 96, k=1, leaky=False)
        self.conv1x1_2 = Conv(c5, 96, k=1, leaky=False)

        # neck
        self.neck = nn.Sequential(
            SPP(),
            Conv(96*4, 96, k=1),
            DWBottleneckCSP(96, 96, n=3, shortcut=False)
        )

         # head
        self.head_conv_0 = Conv(96, 96, k=1)
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_csp_0 = DWBottleneckCSP(96, 96, n=3, shortcut=False)

        # P3/8-small
        self.head_conv_1 = Conv(96, 96, k=1)
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_csp_1 = DWBottleneckCSP(96, 96, n=3, shortcut=False)

        # P4/16-medium
        self.head_conv_2 = nn.Sequential(
            Conv(96, 96, k=3, p=1, s=2),
            Conv(96, 96, k=1)
        )
        self.head_csp_2 = DWBottleneckCSP(96, 96, n=3, shortcut=False)

        # P8/32-large
        self.head_conv_3 = nn.Sequential(
            Conv(96, 96, k=3, p=1, s=2),
            Conv(96, 96, k=1)
        )
        self.head_csp_3 = DWBottleneckCSP(96, 96, n=3, shortcut=False)

        # det conv
        self.head_det_1 = nn.Conv2d(96, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        self.head_det_2 = nn.Conv2d(96, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        self.head_det_3 = nn.Conv2d(96, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)


    def create_grid(self, input_size):
        total_grid_xy = []
        total_stride = []
        total_anchor_wh = []
        w, h = input_size, input_size
        for ind, s in enumerate(self.stride):
            # generate grid cells
            ws, hs = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
            grid_xy = grid_xy.view(1, hs*ws, 1, 2)

            # generate stride tensor
            stride_tensor = torch.ones([1, hs*ws, self.num_anchors, 2]) * s

            # generate anchor_wh tensor
            anchor_wh = self.anchor_size[ind].repeat(hs*ws, 1, 1)

            total_grid_xy.append(grid_xy)
            total_stride.append(stride_tensor)
            total_anchor_wh.append(anchor_wh)

        total_grid_xy = torch.cat(total_grid_xy, dim=1).to(self.device)
        total_stride = torch.cat(total_stride, dim=1).to(self.device)
        total_anchor_wh = torch.cat(total_anchor_wh, dim=0).to(self.device).unsqueeze(0)

        return total_grid_xy, total_stride, total_anchor_wh


    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)


    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [x, y, w, h]
        """
        # b_x = sigmoid(tx)*2-1 + gride_x,  b_y = sigmoid(ty)*2-1 + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        c_xy_pred = (torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell) * self.stride_tensor
        # c_xy_pred = (torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell) * self.stride_tensor
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        b_wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchors_wh
        # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
        xywh_pred = torch.cat([c_xy_pred, b_wh_pred], -1).view(B, HW*ab_n, 4)

        return xywh_pred


    def decode_boxes(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W, anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [B, H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)
        
        return x1y1x2y2_pred


    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def diou_nms(self, dets, scores):
        """"Pure Python DIoU-NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            # compute diou
            # # compute the length of diagonal line
            x1_, x2_ = x1[i].repeat(len(order[1:])), x2[i:i+1].repeat(len(order[1:]))
            y1_, y2_ = y1[i].repeat(len(order[1:])), y2[i:i+1].repeat(len(order[1:]))
            x1234 = np.stack([x1_, x2_, x1[order[1:]], x2[order[1:]]], axis=1)
            y1234 = np.stack([y1_, y2_, y1[order[1:]], y2[order[1:]]], axis=1)

            C = np.sqrt((np.max(x1234, axis=1) - np.min(x1234, axis=1))**2 + \
                        (np.max(y1234, axis=1) - np.min(y1234, axis=1))**2)
            # # compute the distance between two center point
            # # # points-1
            points_1_x = (x1_ + x2_) / 2.
            points_1_y = (y1_ + y2_) / 2.
            # # points-2
            points_2_x = (x1[order[1:]] + x2[order[1:]]) / 2.
            points_2_y = (y1[order[1:]] + y2[order[1:]]) / 2.
            D = np.sqrt((points_2_x - points_1_x)**2 + (points_2_y - points_1_y)**2)

            lens = D**2 / (C**2 + 1e-20)
            diou = iou - lens

            ovr = diou                
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, all_local, all_conf):
        """
        bbox_pred: (HxW*anchor_n, 4), bsize = 1
        prob_pred: (HxW*anchor_n, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms_processor(c_bboxes, c_scores) # self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bbox_pred, scores, cls_inds


    def forward(self, x, target=None):
        # backbone
        c3, c4, c5 = self.backbone(x)
        p3 = self.conv1x1_0(c3)
        p4 = self.conv1x1_1(c4)
        p5 = self.conv1x1_2(c5)

        # neck
        p5 = self.neck(p5)

        # FPN + PAN
        # head
        p6 = self.head_conv_0(p5)
        p7 = self.head_upsample_0(p6)   # s32->s16
        p8 = p7 + p4
        p9 = self.head_csp_0(p8)
        # P3/8
        p10 = self.head_conv_1(p9)
        p11 = self.head_upsample_1(p10)   # s16->s8
        p12 = p11 + p3
        p13 = self.head_csp_1(p12)  # to det
        # p4/16
        p14 = self.head_conv_2(p13)
        p15 = p14 + p10
        p16 = self.head_csp_2(p15)  # to det
        # p5/32
        p17 = self.head_conv_3(p16)
        p18 = p17 + p6
        p19 = self.head_csp_3(p18)  # to det

        # det
        pred_s = self.head_det_1(p13)
        pred_m = self.head_det_2(p16)
        pred_l = self.head_det_3(p19)

        preds = [pred_s, pred_m, pred_l]
        total_conf_pred = []
        total_cls_pred = []
        total_txtytwth_pred = []
        B = HW = 0
        for pred in preds:
            B_, abC_, H_, W_ = pred.size()

            # [B, anchor_n * C, H, W] -> [B, H, W, anchor_n * C] -> [B, H*W, anchor_n*C]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B_, H_*W_, abC_)

            # Divide prediction to obj_pred, xywh_pred and cls_pred   
            # [B, H*W*anchor_n, 1]
            conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B_, H_*W_*self.num_anchors, 1)
            # [B, H*W*anchor_n, num_cls]
            cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B_, H_*W_*self.num_anchors, self.num_classes)
            # [B, H*W*anchor_n, 4]
            txtytwth_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()

            total_conf_pred.append(conf_pred)
            total_cls_pred.append(cls_pred)
            total_txtytwth_pred.append(txtytwth_pred)
            B = B_
            HW += H_*W_
        
        conf_pred = torch.cat(total_conf_pred, dim=1)
        cls_pred = torch.cat(total_cls_pred, dim=1)
        txtytwth_pred = torch.cat(total_txtytwth_pred, dim=1)
        
        # train
        if self.trainable:
            txtytwth_pred = txtytwth_pred.view(B, HW, self.num_anchors, 4)
            
            # 从txtytwth预测中解算出x1y1x2y2坐标
            x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.input_size).view(-1, 4)
            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)
            # 计算pred box与gt box之间的IoU
            iou_pred = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)

            # gt conf，这一操作是保证iou不会回传梯度
            with torch.no_grad():
                gt_conf = iou_pred.clone()

            # 我们讲pred box与gt box之间的iou作为objectness的学习目标. 
            # [obj, cls, txtytwth, scale_weight, x1y1x2y2] -> [conf, obj, cls, txtytwth, scale_weight]
            target = torch.cat([gt_conf, target[:, :, :7]], dim=2)
            txtytwth_pred = txtytwth_pred.view(B, -1, 4)

            # 计算loss
            conf_loss, cls_loss, bbox_loss, iou_loss = tools.loss(pred_conf=conf_pred, 
                                                                  pred_cls=cls_pred,
                                                                  pred_txtytwth=txtytwth_pred,
                                                                  pred_iou=iou_pred,
                                                                  label=target
                                                                  )

            return conf_loss, cls_loss, bbox_loss, iou_loss 
                       
        # test
        else:
            txtytwth_pred = txtytwth_pred.view(B, HW, self.num_anchors, 4)
            with torch.no_grad():
                # batch size = 1
                # 测试时，笔者默认batch是1，
                # 因此，我们不需要用batch这个维度，用[0]将其取走。
                # [B, H*W*num_anchor, 1] -> [H*W*num_anchor, 1]
                conf_pred = torch.sigmoid(conf_pred)[0]
                # [B, H*W*num_anchor, 4] -> [H*W*num_anchor, 4]
                bboxes = torch.clamp((self.decode_boxes(txtytwth_pred) / self.input_size)[0], 0., 1.)
                # [B, H*W*num_anchor, C] -> [H*W*num_anchor, C], 
                scores = torch.softmax(cls_pred[0, :, :], dim=1) * conf_pred

                # 将预测放在cpu处理上，以便进行后处理
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # 后处理
                bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

                return bboxes, scores, cls_inds
