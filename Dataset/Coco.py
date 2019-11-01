import torch
from pycocotools.coco import COCO
import os.path as ops
from PIL import Image
import numpy as np
import random



class Dataset():
    def __init__(self, data_path, mode="Train", transform=None):
        assert  "Train" or "Val" in mode
        self.annotation_train = ops.join(data_path, "annotations", "person_keypoints_train2017.json")
        self.annotation_val = ops.join(data_path, "annotations", "person_keypoints_val2017.json")
        self.train_images = ops.join(data_path, "train2017")
        self.val_images = ops.join(data_path, "val2017")
        self.mode = mode
        self.transform = transform

        self.num_keypoints = 17
        self.joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Thorax', 'Pelvis', "Head", "Spine")

        self.lshoulder_idx = self.joints_name.index('L_Shoulder')
        self.rshoulder_idx = self.joints_name.index('R_Shoulder')

        self.lhip_idx = self.joints_name.index('L_Hip')
        self.rhip_idx = self.joints_name.index('R_Hip')

        self.rear_idx = self.joints_name.index('R_Ear')
        self.lear_idx = self.joints_name.index('L_Ear')


        # this is the H36m keypoints ordering
        self.h36m_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Spine', 'Thorax',
                          'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist',  'R_Shoulder', 'R_Elbow', 'R_Wrist')


        # keypoints ordering to transform coco 2 h36m
        self.coco_2_h36m = [self.joints_name.index('Pelvis'), self.joints_name.index('R_Hip'), self.joints_name.index('R_Knee'),
                            self.joints_name.index('R_Ankle'), self.joints_name.index('L_Hip'), self.joints_name.index('L_Knee'),
                            self.joints_name.index('L_Ankle'), self.joints_name.index('Spine'), self.joints_name.index('Thorax'),
                            self.joints_name.index('Nose'), self.joints_name.index('Head'), self.joints_name.index('L_Shoulder'),
                            self.joints_name.index('L_Elbow'), self.joints_name.index('L_Wrist'), self.joints_name.index('R_Shoulder'),
                            self.joints_name.index('R_Elbow'), self.joints_name.index('R_Wrist')]

        self.data = self.load_data(self.mode)

    def __getitem__(self, item):
        data = self.data[item]
        img = Image.open(data["imgpath"])
        img_id = torch.tensor(data["image_id"])

        gts = data["bbox"].copy()
        gts_length = len(gts)

        gts = torch.as_tensor(gts, dtype=torch.float32)
        labels = torch.ones((gts_length,), dtype=torch.int64)

        area = (gts[:, 3] - gts[:, 1]) * (gts[:, 2] - gts[:, 0])
        iscrowd = torch.zeros((gts_length, ), dtype=torch.int64)
        keypoints = torch.as_tensor(data['joints'], dtype=torch.float32)

        target = dict()
        target['boxes'] = gts
        target['labels'] = labels
        target['image_id'] = img_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['keypoints'] = keypoints.reshape(-1, 17, 3)

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def load_data(self, mode="Train"):
        assert "Train" or "Val" in mode

        if mode is "Train":
            coco = COCO(self.annotation_train)
            path_image = self.train_images
        else:
            coco = COCO(self.annotation_val)
            path_image = self.val_images

        data = []
        for aid in coco.anns.keys():
            ann = coco.anns[aid]
            if ann['image_id'] not in coco.imgs:
                continue

            if (ann['image_id'] not in coco.imgs) or ann['iscrowd'] or (np.sum(ann['keypoints'][2::3]) == 0) or (ann['num_keypoints'] == 0):
                continue

            imgname = path_image + "/" + coco.imgs[ann['image_id']]['file_name']
            x, y, x1, y1 = ann['bbox']
            joints = [ann['keypoints']]
            data_solo = dict(image_id=ann['image_id'], imgpath=imgname, bbox=[[x, y, x1 + x, y1 + y]], joints=[joints], score=1)
            data.append(data_solo)
        return data

    def __len__(self):
        return len(self.data)

    def transform_coco_2_h36m(self, keypoints):
        """
        Transform coco keypoints convention to H36M keypoints convention
        :param keypoints:
        :return:
        """
        # TODO thorax[2] = joint_img[self.lshoulder_idx,2] * joint_img[self.rshoulder_idx,2] add confidence ?
        hip = keypoints[self.lhip_idx] + keypoints[self.rhip_idx] / 2  # mid hip also called pelvis
        thorax = keypoints[self.lshoulder_idx] + keypoints[self.rshoulder_idx] / 2
        head = keypoints[self.lear_idx] + keypoints[self.rear_idx] / 2
        spine = thorax + hip / 2

        # 'Thorax', 'Pelvis', "Head", "Spine"
        spine_hip_thorax_head = np.array((thorax, hip, head, spine))
        keypoints = np.concatenate((keypoints, spine_hip_thorax_head))
        keypoints = keypoints[self.coco_2_h36m]

        return keypoints





"""
load_data = Dataset(data_path="../Data", mode="Train")
item = load_data.__getitem__(15)
image = np.array(item[0])

bboxes = item[1]['boxes'][0]
keypoints = item[1]['keypoints']
print(keypoints)
print(bboxes)
cv2.rectangle(image, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), (255, 255, 255), 2)

cv2.imshow("this is my girl", image)
cv2.waitKey(10000)
"""


