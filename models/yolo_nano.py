import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv, SPP
from backbone import *
import numpy as np
import tools


class YOLONano(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.50, anchor_size=None, backbone='1.0x', diou_nms=False):
        super(YOLONano, self).__init__()
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
        self.anchor_number = self.anchor_size.size(1)

        self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=device).float()

        if self.bk == '0.5x':
            # use shufflenetv2_0.5x as backbone
            print('Use backbone: shufflenetv2_0.5x')
            self.backbone = shufflenetv2(model_size=self.bk, pretrained=trainable)
            width = 0.4138
        elif self.bk == '1.0x':
            # use shufflenetv2_1.0x as backbone
            print('Use backbone: shufflenetv2_1.0x')
            self.backbone = shufflenetv2(model_size=self.bk, pretrained=trainable)
            width = 1.0
        else:
            print("For YOLO-Nano, we only support <0.5x, 1.0x> as our backbone !!")
            exit(0)

        # # SPP
        # self.spp = nn.Sequential(
        #     Conv(int(464*width), int(232*width), k=1),
        #     SPP(),
        #     Conv(int(232*width)*4, int(464*width), k=1)
        # )

        # FPN+PAN
        self.conv1x1_0 = Conv(int(116*width), 96, k=1)
        self.conv1x1_1 = Conv(int(232*width), 96, k=1)
        self.conv1x1_2 = Conv(int(464*width), 96, k=1)

        self.smooth_0 = Conv(96, 96, k=3, p=1)
        self.smooth_1 = Conv(96, 96, k=3, p=1)
        self.smooth_2 = Conv(96, 96, k=3, p=1)
        self.smooth_3 = Conv(96, 96, k=3, p=1)

        # det head
        self.head_det_1 = nn.Sequential(
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            nn.Conv2d(96, self.anchor_number * (1 + self.num_classes + 4), 1)
        )
        self.head_det_2 = nn.Sequential(
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            nn.Conv2d(96, self.anchor_number * (1 + self.num_classes + 4), 1)
        )
        self.head_det_3 = nn.Sequential(
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            nn.Conv2d(96, self.anchor_number * (1 + self.num_classes + 4), 1)
        )


    def create_grid(self, input_size):
        total_grid_xy = []
        total_stride = []
        total_anchor_wh = []
        w, h = input_size[1], input_size[0]
        for ind, s in enumerate(self.stride):
            # generate grid cells
            ws, hs = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
            grid_xy = grid_xy.view(1, hs*ws, 1, 2)

            # generate stride tensor
            stride_tensor = torch.ones([1, hs*ws, self.anchor_number, 2]) * s

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
        self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()


    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [x, y, w, h]
        """
        # b_x = sigmoid(tx)*2-1 + gride_x,  b_y = sigmoid(ty)*2-1 + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        c_xy_pred = (torch.sigmoid(txtytwth_pred[:, :, :, :2]) * 2.0 - 1.0 + self.grid_cell) * self.stride_tensor
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


    def postprocess(self, all_local, all_conf, exchange=True, im_shape=None):
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

        if im_shape != None:
            # clip
            bbox_pred = self.clip_boxes(bbox_pred, im_shape)

        return bbox_pred, scores, cls_inds


    def forward(self, x, target=None):
        # backbone
        c3, c4, c5 = self.backbone(x)

        # # neck
        # c5 = self.spp(c5)

        p3 = self.conv1x1_0(c3)
        p4 = self.conv1x1_1(c4)
        p5 = self.conv1x1_2(c5)

        # FPN
        p4 = self.smooth_0(p4 + F.interpolate(p5, scale_factor=2.0))
        p3 = self.smooth_1(p3 + F.interpolate(p4, scale_factor=2.0))

        # PAN
        p4 = self.smooth_2(p4 + F.interpolate(p3, scale_factor=0.5))
        p5 = self.smooth_3(p5 + F.interpolate(p4, scale_factor=0.5))

        # det head
        pred_s = self.head_det_1(p3)
        pred_m = self.head_det_2(p4)
        pred_l = self.head_det_3(p5)

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
            conf_pred = pred[:, :, :1 * self.anchor_number].contiguous().view(B_, H_*W_*self.anchor_number, 1)
            # [B, H*W*anchor_n, num_cls]
            cls_pred = pred[:, :, 1 * self.anchor_number : (1 + self.num_classes) * self.anchor_number].contiguous().view(B_, H_*W_*self.anchor_number, self.num_classes)
            # [B, H*W*anchor_n, 4]
            txtytwth_pred = pred[:, :, (1 + self.num_classes) * self.anchor_number:].contiguous()

            total_conf_pred.append(conf_pred)
            total_cls_pred.append(cls_pred)
            total_txtytwth_pred.append(txtytwth_pred)
            B = B_
            HW += H_*W_
        
        conf_pred = torch.cat(total_conf_pred, 1)
        cls_pred = torch.cat(total_cls_pred, 1)
        txtytwth_pred = torch.cat(total_txtytwth_pred, 1) #.view(B, -1, 4)
        
        # train
        if self.trainable:
            txtytwth_pred = txtytwth_pred.view(B, HW, self.anchor_number, 4)
            
            # x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.scale_torch).view(-1, 4)
            # x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)

            # # compute iou
            # iou_pred = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)

            # txtytwth_pred = txtytwth_pred.view(B, -1, 4)
            # # compute loss
            # conf_loss, cls_loss, bbox_loss, total_loss = tools.loss(pred_conf=conf_pred,
            #                                                         pred_cls=cls_pred,
            #                                                         pred_txtytwth=txtytwth_pred,
            #                                                         label=target,
            #                                                         num_classes=self.num_classes
            #                                                         )
            
            # decode bbox, and remember to cancel its grad since we set iou as the label of objectness.
            with torch.no_grad():
                x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.scale_torch).view(-1, 4)

            txtytwth_pred = txtytwth_pred.view(B, -1, 4)

            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)

            # compute iou
            iou = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)

            # we set iou between pred bbox and gt bbox as conf label. 
            # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
            target = torch.cat([iou, target[:, :, :7]], dim=2)

            conf_loss, cls_loss, bbox_loss, total_loss = tools.loss(pred_conf=conf_pred, 
                                                                        pred_cls=cls_pred,
                                                                        pred_txtytwth=txtytwth_pred,
                                                                        label=target,
                                                                        num_classes=self.num_classes)


            return conf_loss, cls_loss, bbox_loss, total_loss

        # test
        else:
            txtytwth_pred = txtytwth_pred.view(B, HW, self.anchor_number, 4)
            with torch.no_grad():
                # batch size = 1                
                all_obj = torch.sigmoid(conf_pred)[0]           # 0 is because that these is only 1 batch.
                all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)
                all_class = (torch.softmax(cls_pred[0, :, :], dim=1) * all_obj)
                # all_class = (torch.sigmoid(cls_pred[0, :, :]) * all_obj)
                # separate box pred and class conf
                all_obj = all_obj.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()
                all_bbox = all_bbox.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

                # print(len(all_boxes))
                return bboxes, scores, cls_inds
