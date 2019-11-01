import torch
import torchvision
from torch import nn
import os.path as osp
from .Models.keypoints_rcnn import KeypointRCNN
from .Layers.Light_Head import LightHead
from .Models.ResNet import resnet
from torchvision.models.detection.rpn import AnchorGenerator


COCO_PERSON_KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    ("left_hip", "right_hip", (255, 102, 0)),
    ("left_hip", "left_knee", (255, 255, 77)),
    ("right_hip", "right_knee", (153, 255, 204)),
    ("left_knee", "left_ankle", (191, 255, 128)),
    ("right_knee", "right_ankle", (255, 195, 77)),
]


class KeypointInference():
    def __init__(self, pretrained_model_path, c_out=10):
        self.pretrained_model_path = pretrained_model_path
        backbone = resnet(pretrained=True, is_features=True)
        backbone = LightHead(512, backbone, "S", 10)
        backbone.out_channels = c_out

        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)

        keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=14, sampling_ratio=2)
        self.model = KeypointRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, keypoint_roi_pool=keypoint_roi_pooler, min_size=700, max_size=1100,
                                  box_positive_fraction=0.1,  box_score_thresh=0.15, box_nms_thresh=0.05)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.load_state()
        self.model.eval()

    def load_state(self):
        self.model.load_state_dict(torch.load(self.pretrained_model_path))

    def forward(self, images):
        images = [image.to(self.device) for image in images]
        predictions = self.model(images)
        return predictions

