import torchvision
import torch
from Models.keypoints_rcnn import KeypointRCNN
from torch.utils.data import DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from Utils.Engine import train_one_epoch, evaluate
from Utils import Transforms as T
from Dataset.Coco import Dataset
from Models.ResNet import resnet
from Models.Xception import xception
from Models.HardNet import HarDNet
from Models.DetNas import create_network
from Models.Shuffle import shufflenet
from EfficientNet.mobilenetv3 import mobilenetv3_100


c_out_is_small = True


if c_out_is_small:
    c_out = 10
else:
    c_out = 10 * 7 * 7



def get_transform(train):
    transforms = []

    if train:
        transforms.append(T.ColorJitter(brightness=2, contrast=0.5, hue=0.1))
        transforms.append(T.ToTensor())
        transforms.append(T.RandomHorizontalFlip(0.5))
    else:
        transforms.append(T.ToTensor())
    return T.Compose(transforms)

train_dataset = Dataset(data_path="./Data", mode="Train", transform=get_transform(train=True))
validation_dataset = Dataset(data_path="./Data", mode="val", transform=get_transform(train=False))

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(validation_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)


class LightHead(torch.nn.Module):
    def __init__(self, in_, backbone, mode="S"):
        super(LightHead, self).__init__()
        self.backbone = backbone
        if mode == "L":
            self.out_mode = 256
        else:
            self.out_mode = 64
        self.conv1 = torch.nn.Conv2d(in_channels=in_, out_channels=self.out_mode, kernel_size=(15, 1), stride=1,
                                     padding=(7, 0), bias=True)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.out_mode, out_channels=c_out, kernel_size=(1, 15),  stride=1,
                                     padding=(0, 7), bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=in_, out_channels=self.out_mode, kernel_size=(15, 1), stride=1, padding=(7, 0), bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=self.out_mode, out_channels=c_out, kernel_size=(1, 15), stride=1,
                                     padding=(0, 7), bias=True)

    def forward(self, input):
        x_backbone = self.backbone(input)
        x = self.conv1(x_backbone)
        x = self.relu(x)
        x = self.conv2(x)
        x_relu_2 = self.relu(x)

        x = self.conv3(x_backbone)
        x = self.relu(x)
        x = self.conv4(x)
        x_relu_4 = self.relu(x)

        return x_relu_2 + x_relu_4


# load a pre-trained model for classification and return
# only the features
backbone = mobilenetv3_100(pretrained=True)
backbone = LightHead(1280, backbone=backbone)

backbone.out_channels = c_out

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)

keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=14, sampling_ratio=2)

model = KeypointRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, keypoint_roi_pool=keypoint_roi_pooler, min_size=800, max_size=1200)
#model.load_state_dict(torch.load('./keypoints_rcnn_.pth'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005) # add one zero


# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

# let's train it for 10 epochs
num_epochs = 30


for epoch in range(0, num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, test_loader, device=device)
    torch.save(model.state_dict(), "keypoints_rcnn_" + ".pth")


"""
17  3.6507, 0.0257

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.428
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.772
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.423
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.318
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.229
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.512
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.433
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.595


IoU metric: keypoints
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.529
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.743
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.573
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.378
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.648
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.840
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.696
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.524
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.733
"""

