import numpy as np
from data import *
import torch.nn as nn
import torch.nn.functional as F
import math

# We use ignore thresh to decide which anchor box can be kept.
ignore_thresh = IGNORE_THRESH


# new add func.
class MSELoss(nn.Module):
    def __init__(self,  weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets, mask):
        # We ignore those whose tarhets == -1.0. 
        pos_id = (mask==1.0).float()
        neg_id = (mask==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss


def generate_anchor(input_size, stride, anchor_scale, anchor_aspect):
    """
        The function is used to design anchor boxes by ourselves as long as you provide the scale and aspect of anchor boxes.
        Input:
            input_size : list -> the image resolution used in training stage and testing stage.
            stride : int -> the downSample of the CNN, such as 32, 64 and so on.
            anchor_scale : list -> it contains the area ratio of anchor boxes. For example, anchor_scale = [0.1, 0.5]
            anchor_aspect : list -> it contains the aspect ratios of anchor boxes for various anchor area.
                            For example, anchor_aspect = [[1.0, 2.0], [3.0, 1/3]]. And len(anchor_aspect) must 
                            be equal to len(anchor_scale).
        Output:
            total_anchor_size : list -> [[h_1, w_1], [h_2, w_2], ..., [h_n, w_n]].
    """
    assert len(anchor_scale) == len(anchor_aspect)
    h, w = input_size
    hs, ws = h // stride, w // stride
    S_fmap = hs * ws
    total_anchor_size = []
    for ab_scale, aspect_ratio in zip(anchor_scale, anchor_aspect):
        for a in aspect_ratio:
            S_ab = S_fmap * ab_scale
            ab_w = np.floor(np.sqrt(S_ab))
            ab_h =ab_w * a
            total_anchor_size.append([ab_w, ab_h])
    return total_anchor_size


def compute_iou(anchor_boxes, gt_box):
    """
    Input:
        anchor_boxes : ndarray -> [[c_x_s, c_y_s, anchor_w, anchor_h], ..., [c_x_s, c_y_s, anchor_w, anchor_h]].
        gt_box : ndarray -> [c_x_s, c_y_s, anchor_w, anchor_h].
    Output:
        iou : ndarray -> [iou_1, iou_2, ..., iou_m], and m is equal to the number of anchor boxes.
    """
    # compute the iou between anchor box and gt box
    # First, change [c_x_s, c_y_s, anchor_w, anchor_h] ->  [xmin, ymin, xmax, ymax]
    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # xmin
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # ymin
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # xmax
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # ymax
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]
    
    # gt_box : 
    # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily. 
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2 # xmin
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2 # ymin
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2 # xmax
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2 # ymin
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # Then we compute IoU between anchor_box and gt_box
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U
    
    return IoU


def set_anchors(anchor_size):
    """
    Input:
        anchor_size : list -> [[h_1, w_1], [h_2, w_2], ..., [h_n, w_n]].
    Output:
        anchor_boxes : ndarray -> [[0, 0, anchor_w, anchor_h],
                                   [0, 0, anchor_w, anchor_h],
                                   ...
                                   [0, 0, anchor_w, anchor_h]].
    """
    anchor_number = len(anchor_size)
    anchor_boxes = np.zeros([anchor_number, 4])
    for index, size in enumerate(anchor_size): 
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])
    
    return anchor_boxes


# new add func.
def assign_labels(gt_tensor, stride, batch_index, s_indx, ab_ind, box_w, box_h, anchor_w, anchor_h, gt_class, grid_x, grid_y, cx_s, cy_s, iou, xmin, ymin, xmax, ymax, w, h):
    # gt labels
    if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
        tx = cx_s - grid_x
        ty = cy_s - grid_y
        tw = np.log(box_w / anchor_w)
        th = np.log(box_h / anchor_h)
        weight = 2.0 - (box_w / w) * (box_h / h) # 2.0 - iou 

        gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
        gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
        gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([tx, ty, tw, th])
        gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
        gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = np.array([xmin, ymin, xmax, ymax])

    return gt_tensor


# new add func.
def multi_gt_creator_new(input_size, strides, label_lists=[], anchor_size=None):
    """creator multi scales gt"""
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h, w = input_size
    num_scale = len(strides)
    gt_tensor = []

    anchor_number = len(anchor_size) // num_scale
    
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h//s, w//s, anchor_number, 1+1+4+1+4]))

    # generate gt datas
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            gt_class = int(gt_label[-1])
            xmin, ymin, xmax, ymax = gt_label[:-1]
            # compute the center, width and height
            cx = (xmax + xmin) / 2 * w
            cy = (ymax + ymin) / 2 * h
            box_w = (xmax - xmin) * w
            box_h = (ymax - ymin) * h

            if box_w < 1. or box_h < 1.:
                # This is not a valid gt data.
                continue    

            # compute the IoU
            anchor_boxes = set_anchors(anchor_size)
            gt_box = np.array([[0, 0, box_w, box_h]])
            iou = compute_iou(anchor_boxes, gt_box)

            # We only consider those anchor boxes whose IoU is more than ignore thresh,
            iou_mask = (iou > ignore_thresh)

            # The grid cell：
            # 
            #   |    |
            # — p0 — p1 —
            #   |    |
            # — p2 — p3 —
            #   |    |
            #
            if iou_mask.sum() == 0:
                # We assign the anchor box with highest IoU score.
                index = np.argmax(iou)
                # s_indx, ab_ind = index // num_scale, index % num_scale
                s_indx = index // anchor_number
                ab_ind = index - s_indx * anchor_number
                # get the corresponding stride
                s = strides[s_indx]
                # get the corresponding anchor box
                anchor_w, anchor_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                # compute the gride cell location
                cx_s = cx / s
                cy_s = cy / s
                # p0
                grid_x_c_0 = int(cx_s)
                grid_y_c_0 = int(cy_s)
                gt_tensor = assign_labels(gt_tensor, s, batch_index, s_indx, ab_ind, 
                                        box_w, box_h, anchor_w, anchor_h, 
                                        gt_class, 
                                        grid_x_c_0, grid_y_c_0, 
                                        cx_s, cy_s, 
                                        iou, 
                                        xmin, ymin, xmax, ymax, w, h
                                        )

                # p1
                grid_x_c_1 = grid_x_c_0 + 1
                grid_y_c_1 = grid_y_c_0
                gt_tensor = assign_labels(gt_tensor, s, batch_index, s_indx, ab_ind, 
                                        box_w, box_h, anchor_w, anchor_h, 
                                        gt_class, 
                                        grid_x_c_1, grid_y_c_1, 
                                        cx_s, cy_s, 
                                        iou, 
                                        xmin, ymin, xmax, ymax, w, h
                                        )

                # p2
                grid_x_c_2 = grid_x_c_0
                grid_y_c_2 = grid_y_c_0 + 1
                gt_tensor = assign_labels(gt_tensor, s, batch_index, s_indx, ab_ind, 
                                        box_w, box_h, anchor_w, anchor_h, 
                                        gt_class, 
                                        grid_x_c_2, grid_y_c_2, 
                                        cx_s, cy_s, 
                                        iou, 
                                        xmin, ymin, xmax, ymax, w, h
                                        )

                # p3
                grid_x_c_3 = grid_x_c_0 + 1
                grid_y_c_3 = grid_y_c_0 + 1
                gt_tensor = assign_labels(gt_tensor, s, batch_index, s_indx, ab_ind, 
                                        box_w, box_h, anchor_w, anchor_h, 
                                        gt_class, 
                                        grid_x_c_3, grid_y_c_3, 
                                        cx_s, cy_s, 
                                        iou, 
                                        xmin, ymin, xmax, ymax, w, h
                                        )
            
            else:
                # There are more than one anchor boxes whose IoU are higher than ignore thresh.               
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        # s_indx, ab_ind = index // num_scale, index % num_scale
                        s_indx = index // anchor_number
                        ab_ind = index - s_indx * anchor_number
                        # get the corresponding stride
                        s = strides[s_indx]
                        # get the corresponding anchor box
                        anchor_w, anchor_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                        # compute the gride cell location
                        cx_s = cx / s
                        cy_s = cy / s

                        # p0
                        grid_x_c_0 = int(cx_s)
                        grid_y_c_0 = int(cy_s)
                        gt_tensor = assign_labels(gt_tensor, s, batch_index, s_indx, ab_ind, 
                                                box_w, box_h, anchor_w, anchor_h, 
                                                gt_class, 
                                                grid_x_c_0, grid_y_c_0, 
                                                cx_s, cy_s, 
                                                iou, 
                                                xmin, ymin, xmax, ymax, w, h
                                                )

                        # p1
                        grid_x_c_1 = grid_x_c_0 + 1
                        grid_y_c_1 = grid_y_c_0
                        gt_tensor = assign_labels(gt_tensor, s, batch_index, s_indx, ab_ind, 
                                                box_w, box_h, anchor_w, anchor_h, 
                                                gt_class, 
                                                grid_x_c_1, grid_y_c_1, 
                                                cx_s, cy_s, 
                                                iou, 
                                                xmin, ymin, xmax, ymax, w, h
                                                )

                        # p2
                        grid_x_c_2 = grid_x_c_0
                        grid_y_c_2 = grid_y_c_0 + 1
                        gt_tensor = assign_labels(gt_tensor, s, batch_index, s_indx, ab_ind, 
                                                box_w, box_h, anchor_w, anchor_h, 
                                                gt_class, 
                                                grid_x_c_2, grid_y_c_2, 
                                                cx_s, cy_s, 
                                                iou, 
                                                xmin, ymin, xmax, ymax, w, h
                                                )

                        # p3
                        grid_x_c_3 = grid_x_c_0 + 1
                        grid_y_c_3 = grid_y_c_0 + 1
                        gt_tensor = assign_labels(gt_tensor, s, batch_index, s_indx, ab_ind, 
                                                box_w, box_h, anchor_w, anchor_h, 
                                                gt_class, 
                                                grid_x_c_3, grid_y_c_3, 
                                                cx_s, cy_s, 
                                                iou, 
                                                xmin, ymin, xmax, ymax, w, h
                                                )
                    
    gt_tensor = [gt.reshape(batch_size, -1, 1+1+4+1+4) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, 1)
    
    return gt_tensor


def multi_gt_creator(input_size, strides, label_lists=[], anchor_size=None):
    """creator multi scales gt"""
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h, w = input_size
    num_scale = len(strides)
    gt_tensor = []

    # generate gt datas
    all_anchor_size = anchor_size # get_total_anchor_size(multi_level=True, name=name, version=version)
    anchor_number = len(all_anchor_size) // num_scale
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h//s, w//s, anchor_number, 1+1+4+1+4]))
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            gt_class = int(gt_label[-1])
            xmin, ymin, xmax, ymax = gt_label[:-1]
            # compute the center, width and height
            c_x = (xmax + xmin) / 2 * w
            c_y = (ymax + ymin) / 2 * h
            box_w = (xmax - xmin) * w
            box_h = (ymax - ymin) * h

            if box_w < 1. or box_h < 1.:
                # print('A dirty data !!!')
                continue    

            # compute the IoU
            anchor_boxes = set_anchors(all_anchor_size)
            gt_box = np.array([[0, 0, box_w, box_h]])
            iou = compute_iou(anchor_boxes, gt_box)

            # We only consider those anchor boxes whose IoU is more than ignore thresh,
            iou_mask = (iou > ignore_thresh)

            if iou_mask.sum() == 0:
                # We assign the anchor box with highest IoU score.
                index = np.argmax(iou)
                # s_indx, ab_ind = index // num_scale, index % num_scale
                s_indx = index // anchor_number
                ab_ind = index - s_indx * anchor_number
                # get the corresponding stride
                s = strides[s_indx]
                # get the corresponding anchor box
                p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                # compute the gride cell location
                c_x_s = c_x / s
                c_y_s = c_y / s
                grid_x = int(c_x_s)
                grid_y = int(c_y_s)
                # compute gt labels
                tx = c_x_s - grid_x
                ty = c_y_s - grid_y
                tw = np.log(box_w / p_w)
                th = np.log(box_h / p_h)
                weight = 2.0 - (box_w / w) * (box_h / h)

                if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = np.array([xmin, ymin, xmax, ymax])
            
            else:
                # There are more than one anchor boxes whose IoU are higher than ignore thresh.                
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        # s_indx, ab_ind = index // num_scale, index % num_scale
                        s_indx = index // anchor_number
                        ab_ind = index - s_indx * anchor_number
                        # get the corresponding stride
                        s = strides[s_indx]
                        # get the corresponding anchor box
                        p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                        # compute the gride cell location
                        c_x_s = c_x / s
                        c_y_s = c_y / s
                        grid_x = int(c_x_s)
                        grid_y = int(c_y_s)
                        # compute gt labels
                        tx = c_x_s - grid_x
                        ty = c_y_s - grid_y
                        tw = np.log(box_w / p_w)
                        th = np.log(box_h / p_h)
                        weight = 2.0 - (box_w / w) * (box_h / h)

                        if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([tx, ty, tw, th])
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = np.array([xmin, ymin, xmax, ymax])
            
    gt_tensor = [gt.reshape(batch_size, -1, 1+1+4+1+4) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, 1)
    
    return gt_tensor


def iou_score(bboxes_a, bboxes_b, batch_size):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    iou = area_i / (area_a + area_b - area_i + 1e-20)

    # size=[B*N] -> [B, N, 1]
    iou = iou.view(batch_size, -1, 1)

    return iou


# new add func.
def loss(pred_conf, pred_cls, pred_txtytwth, label, num_classes):
    obj = 5.0
    noobj = 1.0
    
    # loss func.
    conf_loss_function = MSELoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none') # nn.SmoothL1Loss(reduction='none')
    twth_loss_function = nn.SmoothL1Loss(reduction='none')

    # predictions
    pred_conf = torch.sigmoid(pred_conf[:, :, 0])
    pred_cls = pred_cls.permute(0, 2, 1)
    txty_pred = pred_txtytwth[:, :, :2] # torch.sigmoid(pred_txtytwth[:, :, :2]) * 2.0 - 1.0
    twth_pred = pred_txtytwth[:, :, 2:]

    # gt labels     
    gt_conf = label[:, :, 0].float()
    gt_mask = label[:, :, 1].float()
    gt_cls = label[:, :, 2].long()
    gt_txtytwth = label[:, :, 3:-1].float()
    gt_box_scale_weight = label[:, :, -1]

    batch_size = pred_cls.size(0)
    # objectness loss
    pos_loss, neg_loss = conf_loss_function(pred_conf, gt_conf, gt_mask)
    conf_loss = obj * pos_loss + noobj * neg_loss
    
    # class loss
    cls_loss = torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_mask) / batch_size
    
    # box loss
    txty_loss = torch.sum(torch.sum(txty_loss_function(txty_pred, gt_txtytwth[:, :, :2]), 2) * gt_box_scale_weight * gt_mask) / batch_size
    twth_loss = torch.sum(torch.sum(twth_loss_function(twth_pred, gt_txtytwth[:, :, 2:]), 2) * gt_box_scale_weight * gt_mask) / batch_size

    txtytwth_loss = txty_loss + twth_loss

    total_loss = conf_loss + cls_loss + txtytwth_loss

    return conf_loss, cls_loss, txtytwth_loss, total_loss


if __name__ == "__main__":
    gt_box = np.array([[0.0, 0.0, 10, 10]])
    anchor_boxes = np.array([[0.0, 0.0, 10, 10], 
                             [0.0, 0.0, 4, 4], 
                             [0.0, 0.0, 8, 8], 
                             [0.0, 0.0, 16, 16]
                             ])
    iou = compute_iou(anchor_boxes, gt_box)
    print(iou)

    box1 = torch.FloatTensor([[0,0,8,6], [0, 0, 10, 10]]) / 100.
    box2 = torch.FloatTensor([[2,3,10,9], [100, 100, 10, 10]]) / 100.
    iou = IoU(box1, box2, batch_size=2)
    print('iou: ', iou)
    diou = DIoU(box1, box2, batch_size=2)
    print('diou: ', diou)
    ciou = CIoU(box1, box2, batch_size=2)
    print('ciou: ', ciou)