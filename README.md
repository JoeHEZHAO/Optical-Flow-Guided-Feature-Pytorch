# OFF-Action-Recogniton
Optical Flow Guided Feature for Action Recognition-Pytorch

Paper reference [CVPR2018 OFF for Action Recogniton](https://arxiv.org/pdf/1711.11152.pdf)

## DataSet & Weights Preparation
+ Prepare UCF-101 and HMDB51 dataset follow instruction of [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch);
+ Prepare pretrained UCF-101 weights from tsn-pytorch;

## Network Modules
+ Temporal Segment Network (TSN) & DataLoader, follow [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch)
+ Original Optical Flow Guided Feature, follow [caffe prototxt](https://github.com/kevin-ssy/Optical-Flow-Guided-Feature/blob/master/models/ucf101/rgb_off/1/train.prototxt)
+ Sobel Operator, follow [wiki](https://en.wikipedia.org/wiki/Sobel_operator)
+ Temporal Segment Consensus module in basic_ops.py
+ Translated OFF Network in RGB_OFF.py/Flow_OFF.py
+ Rewrite dataset.py to dataset_off.py, for frame sampling interval consistency, as mentioned in the paper Section 4.2, last paragraph;
+

## Performance

