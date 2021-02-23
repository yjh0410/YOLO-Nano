import json
import tempfile

from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable

from data.cocodataset import *
from data import *


import json
import tempfile

from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable

from data.cocodataset import *
from data import *


class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, data_dir, img_size, device, testset=False, transform=None):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.testset = testset
        if self.testset:
            json_file='image_info_test-dev2017.json'
            name = 'test2017'
        else:
            json_file='instances_val2017.json'
            name='val2017'

        self.dataset = COCODataset(
                                   data_dir=data_dir,
                                   img_size=img_size,
                                   json_file=json_file,
                                   transform=None,
                                   name=name)
        self.dataloader = torch.utils.data.DataLoader(
                                    self.dataset, 
                                    batch_size=1, 
                                    shuffle=False, 
                                    collate_fn=detection_collate,
                                    num_workers=0)
        self.img_size = img_size
        self.transform = transform
        self.device = device

    def preprocess(self, img, height, width):
        # zero padding
        if height > width:
            img_ = np.zeros([height, height, 3])
            delta_w = height - width
            left = delta_w // 2
            img_[:, left:left+width, :] = img
            offset = np.array([[ left / height, 0.,  left / height, 0.]])

        elif height < width:
            img_ = np.zeros([width, width, 3])
            delta_h = width - height
            top = delta_h // 2
            img_[top:top+height, :, :] = img
            offset = np.array([[0.,    top / width, 0.,    top / width]])
        
        else:
            img_ = img
            offset = np.zeros([1, 4])

        return img_, offset

    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))

        # start testing
        for index in range(num_images): # all the data in val2017
            if index % 500 == 0:
                print('[Eval: %d / %d]'%(index, num_images))

            img, id_ = self.dataset.pull_image(index)  # load a batch
            height, width, _ = img.shape

            img, _, _, scale, offset = self.transform(img)

            # img_, offset = self.preprocess(img, height, width)
                
            x = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
            x = x.unsqueeze(0).to(self.device)
            
            id_ = int(id_)
            ids.append(id_)
            with torch.no_grad():
                outputs = model(x)
                bboxes, scores, cls_inds = outputs
                # scale each detection back up to the image
                max_line = max(height, width)
                # map the boxes to input image with zero padding
                bboxes *= max_line
                # map to the image without zero padding
                bboxes -= (offset * max_line)

            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = self.dataset.class_ids[int(cls_inds[i])]
                
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i]) # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score} # COCO json format
                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            if self.testset:
                json.dump(data_dict, open('yolov2_2017.json', 'w'))
                cocoDt = cocoGt.loadRes('yolov2_2017.json')
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, 'w'))
                cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            ap50, ap50_95 = cocoEval.stats[0], cocoEval.stats[1]
            print('ap50_95 : ', ap50_95)
            print('ap50 : ', ap50)

            return ap50, ap50_95
        else:
            return 0, 0
