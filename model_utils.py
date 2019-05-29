import argparse
import time
import os, sys
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
# from BNInception import bninception
from RGB_OFF import bninception_off
import re, deprecated
import pdb

class RBF(nn.Module):

    def __init__(self, num_kernel, input_dim):
        super(RBF, self).__init__()
        self.num_kernel = num_kernel
        self.input_dim = input_dim

        self.center = nn.Parameter(torch.rand(self.num_kernel, self.input_dim) , requires_grad=True)
        self.beta = nn.Parameter(torch.rand(self.num_kernel), requires_grad=True)

    def forward(self, input):
        x= (input-self.center).pow(2).sum(2, keepdim=False).sqrt()
        x = torch.exp(-self.beta.mul(x))
        return x

class topk_crossEntrophy(nn.Module):
    def __init__(self, top_k=0.7):
        super(topk_crossEntrophy, self).__init__()
        self.loss = nn.NLLLoss()
        self.top_k = top_k
        self.softmax = nn.LogSoftmax()

    def forward(self, input, target):
        softmax_result = self.softmax(input)

        loss = torch.autograd.Variable(torch.Tensor(1).zero_()).cuda()
        for idx, row in enumerate(softmax_result):
            gt = target[idx]
            pred = torch.unsqueeze(row, 0)
            gt = torch.unsqueeze(gt, 0)
            cost = self.loss(pred, gt)
            cost = torch.unsqueeze(cost, 0)
            loss = torch.cat((loss, cost), 0)

        loss = loss[1:]
        if self.top_k == 1.0:
            valid_loss = loss

        # import pdb;pdb.set_trace()
        index = torch.topk(loss, int(self.top_k * loss.size()[0]))
        valid_loss = loss[index[1]]

        return torch.mean(valid_loss)

def initNetWeights(net, require_grad=True):
    for m in net.modules():
        if m.requires_grad:
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias:
                    init.constant(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)
                if m.bias:
                    init.constant(m.bias, 0)

def pretrained_bninception():

    sys.path.append('/home/zhufl/Temporal-Residual-Motion-Generation/tsn-pytorch')
    from BNInception_model import bninception
    # net = bninception(num_classes=101, pretrained=None)
    net = bninception()
    checkpoint = torch.load('/home/zhufl/Temporal-Residual-Motion-Generation/tsn-pytorch/ucf101_rgb.pth') # checkpoint only has 416 in length;
    print(len(list(checkpoint)))

    net.fc = nn.Linear(1024, 101)

    count = 0
    base_dict = {}
    for k, v in checkpoint.items():
        count = count + 1
        if 415>count>18:
            print count, k[18:]
            base_dict.setdefault(k[18:], checkpoint[k])
        elif count<19:
            print count, k[11:]
            base_dict.setdefault(k[11:], checkpoint[k])
        # else:
            # '''check un-recovered weights '''
            # print count, k
            # base_dict.setdefault(k[18:], checkpoint[k])

    base_dict.setdefault('fc.weight', checkpoint['base_model.fc-action.1.weight'])
    base_dict.setdefault('fc.bias', checkpoint['base_model.fc-action.1.bias'])

    # For kinetics dataset:
    # base_dict.setdefault('new_fc.weight', checkpoint['base_model.fc_action.1.weight'])
    # base_dict.setdefault('new_fc.bias', checkpoint['base_model.fc_action.1.bias'])

    '''
    mode_state = net.state_dict()
    re_base_dict = {k[11:]:v for k, v in base_dict.items() if k not in mode_state}
    print(re_base_dict.keys())
    base_dict = {k: v for k, v in base_dict.items() if k in mode_state}
    print(len(list(base_dict)))
    print(len(list(re_base_dict)))
    base_dict.update(re_base_dict)
    print(len(list(base_dict)))
    '''

    # mode_state = net.state_dict()
    # re_base_dict = {k:v for k, v in mode_state.items() if k not in base_dict}
    # print(re_base_dict.keys())

    # mode_state.update(base_dict)
    net.load_state_dict(base_dict)
    print("Finish loading the caffemodel ucf101_rgb weights for plain BNInception model")
    return net

# @deprecation.deprecated(deprecated_in="1.0", removed_in="2.0",
#                         current_version=__version__,
#                         details="Use the bar function instead")
# def pretrained_bninception_off():

#     net = bninception_off(101, None)
#     checkpoint = torch.load('/home/zhufl/videoPrediction/rgb_off_reference_split_1.caffemodel.pth')
#     print("Number of parameters recovered from original caffemodel {}".format(len(checkpoint)))

#     # for key,value in checkpoint.items():
#     #     if 'running_var' in key:
#     #         print key

#     # fix inception & fc-action-motion layer name issue
#     for key, value in checkpoint.items():
#         new_key = key.replace("/", "_")
#         new_key = new_key.replace("-", "_")

#         # fix error for copying a param of torch.Size([x]) from checkpoint, where the shape is torch.Size([1, ]) in current model.
#         # Dimension error for all bn layer
#         if '_bn' in key:
#             checkpoint[key] = torch.squeeze(checkpoint[key])
#         checkpoint[new_key] = checkpoint.pop(key)


#     checkpoint['last_linear.weight'] = checkpoint.pop('fc_action.weight')
#     checkpoint['last_linear.bias'] = checkpoint.pop('fc_action.bias')

#     # checkpoint['fc_action_motion.weight'] = checkpoint.pop('fc-action-motion.weight')
#     # checkpoint['fc_action_motion.bias'] = checkpoint.pop('fc-action-motion.bias')
#     # checkpoint['fc_action_motion_28.weight'] = checkpoint.pop('fc-action-motion_28.weight')
#     # checkpoint['fc_action_motion_28.bias'] = checkpoint.pop('fc-action-motion_28.bias')
#     # checkpoint['fc_action_motion_14.weight'] = checkpoint.pop('fc-action-motion_14.weight')
#     # checkpoint['fc_action_motion_14.bias'] = checkpoint.pop('fc-action-motion_14.bias')

#     model_state = net.state_dict()
#     # print("Number of parameters of oringinal model is {}".format(len(model_state)))
#     # for key, value in model_state.items():
#     #     print(key)

#     base_dict = {k:v for k, v in checkpoint.items() if k in model_state}
#     # for key, value in base_dict.items():
#     #     print(key)

#     missing_dict = {k:v for k, v in model_state.items() if k not in checkpoint}
#     for key, value in missing_dict.items():
#         print("Missing {}".format(key))

#     print("Number of parameters loaded from well trained model {}".format(len(base_dict)))

#     model_state.update(base_dict)

#     net.load_state_dict(model_state)
#     print("Load weights and bias from RGB_OFF_caffemodel")
#     return net

def pretrained_bninception_off(batch, num_seg):

    '''
    Load converted rgb_off_ucf101_caffemodel
    bninception_off input: num_class, num_batch/num_crop, num_seg
    '''

    net = bninception_off(101, batch, num_seg)
    checkpoint = torch.load('/home/zhufl/Data2/caffe2pytorch-tsn/converted_rgb_off_ucf101_caffemodel.pth')
    print("Number of parameters recovered from original caffemodel {}".format(len(checkpoint)))

    model_state = net.state_dict()

    '''
        Choose to init motion branch or not
        If only fine-tune last layer, then need to load motion branch;
        If fine-tune whole motion branch, then no need;
    '''
    # base_dict = {k:v for k, v in checkpoint.items() if k in model_state}
    base_dict = {k:v for k, v in checkpoint.items() if 'fc-action' not in k }
    # print(base_dict.keys())
    # import pdb;pdb.set_trace()
    # missing_dict = {k:v for k, v in model_state.items() if k not in base_dict}
    # for key, value in missing_dict.items():
    #     print("Missing motion branch param {}".format(key))

    missing_dict = {k:v for k, v in model_state.items() if k not in checkpoint}
    loading_dict = {k:v for k, v in model_state.items() if k in checkpoint}
    for key, value in missing_dict.items():
        print("Missing {}".format(key))

    for key, value in loading_dict.items():
        print("Loading {}".format(key))

    model_state.update(base_dict)
    net.load_state_dict(model_state)
    print("Finish Load weights and bias from RGB_OFF_caffemodel")
    return net

def fine_tune_bninception_off(batch, num_seg):

    '''
    Loading most recent pre-trained model for fine-tuning;

    bninception_off input: num_class, num_batch/num_crop, num_seg

    motion_spatial_grad is fixed; Should not be able to train;
    '''

    net = bninception_off(101, batch, num_seg)
    checkpoint = torch.load('/home/zhufl/Data2/caffe2pytorch-tsn/converted_rgb_off_ucf101_caffemodel.pth')
    # checkpoint = torch.load('/home/zhufl/Temporal-Residual-Motion-Generation/tsn-pytorch/ucf101_rgb.pth') # checkpoint only has 416 i

    # checkpoint = torch.load('/home/zhufl/videoPrediction/train/' + '2019-01-08_22-42-50.pth')

    print("Number of parameters recovered from original caffemodel {}".format(len(checkpoint)))

    model_state = net.state_dict()

    '''
        Choose to init motion branch or not
        If only fine-tune last layer, then need to load motion branch;
        If fine-tune whole motion branch, then no need;
    '''
    # base_dict = {k:v for k, v in checkpoint.items() if k in model_state}
    # base_dict = {k:v for k, v in checkpoint.items() if 'fc' not in k }
    base_dict = {k:v for k, v in checkpoint.items() if 'motion' not in k }
    # print(base_dict.keys())
    # import pdb;pdb.set_trace()

    missing_dict = {k:v for k, v in model_state.items() if k not in base_dict}
    for key, value in missing_dict.items():
        print("Missing motion branch param {}".format(key))

    # missing_dict = {k:v for k, v in model_state.items() if k not in checkpoint}
    # for key, value in missing_dict.items():
    #     print("Missing {}".format(key))

    model_state.update(base_dict)

    net.load_state_dict(model_state)
    print("Load weights and bias from RGB_OFF_caffemodel")
    return net

def fine_tune_bninception_off_sobel(batch, num_seg):

    '''
    Loading most recent pre-trained model for fine-tuning;

    bninception_off input: num_class, num_batch/num_crop, num_seg

    motion_spatial_grad is fixed; Should not be able to train;
    '''

    from RGB_OFF_v2 import bninception_off
    net = bninception_off(101, batch, num_seg)

    # init all trainable variable
    # initNetWeights(net)
    # model_name = '2019-01-13_00-44-51.pth'
    model_name = '2019-01-14_12-39-41.pth'

    # checkpoint = torch.load('/home/zhufl/Data2/caffe2pytorch-tsn/converted_rgb_off_ucf101_caffemodel.pth')
    checkpoint = torch.load('/home/zhufl/videoPrediction/train/' + model_name)
    print("Number of parameters recovered from original caffemodel {}".format(len(checkpoint)))

    model_state = net.state_dict()

    '''
        Choose to init motion branch or not
        If only fine-tune last layer, then need to load motion branch;
        If fine-tune whole motion branch, then no need;
    '''
    base_dict = {k:v for k, v in checkpoint.items() if k in model_state}
    # base_dict = {k:v for k, v in checkpoint.items() if 'motion' not in k }
    # print(base_dict.keys())
    # import pdb;pdb.set_trace()

    missing_dict = {k:v for k, v in model_state.items() if k not in base_dict}
    for key, value in missing_dict.items():
        print("Missing motion branch param {}".format(key))

    # missing_dict = {k:v for k, v in model_state.items() if k not in checkpoint}
    # for key, value in missing_dict.items():
    #     print("Missing {}".format(key))

    model_state.update(base_dict)

    net.load_state_dict(model_state)
    print("Load weights and bias from " + model_name)
    return net


def selftrained_bninception_off(batch, num_seg, model='RGB_OFF_2019-02-13_15-40-01.pth'):

    '''
        Load self-trained OFF weights;
        bninception_off input: num_class, num_batch/num_crop, num_seg;
    '''

    net = bninception_off(101,batch, num_seg)
    checkpoint = torch.load('/home/zhufl/Temporal-Residual-Motion-Generation/videoPrediction/train/' + model)
    print("Number of parameters recovered from original caffemodel {}".format(len(checkpoint)))

    model_state = net.state_dict()

    base_dict = {k:v for k, v in checkpoint.items() if k in model_state}
    # print(base_dict.keys())

    missing_dict = {k:v for k, v in model_state.items() if k not in checkpoint}
    for key, value in missing_dict.items():
        print("Missing {}".format(key))

    model_state.update(base_dict)

    net.load_state_dict(model_state)
    print("Load weights and bias from '/home/zhufl/videoPrediction/train/" + model)
    return net

def selftrained_bninception_off_sobel(batch, num_seg, model='2019-02-10_12-57-17.pth'):

    '''
        Load self-trained OFF weights;
        bninception_off input: num_class, num_batch/num_crop, num_seg;
    '''

    from RGB_OFF_v2 import bninception_off

    net = bninception_off(101, batch, num_seg)
    net = torch.nn.DataParallel(net, device_ids=[0]).cuda()

    checkpoint = torch.load('/home/zhufl/Temporal-Residual-Motion-Generation/videoPrediction/train/' + model)
    # print("Number of parameters recovered from original caffemodel {}".format(len(checkpoint)))

    # model_state = net.state_dict()

    # base_dict = {k:v for k, v in checkpoint.items() if k in model_state}
    # # print(base_dict.keys())

    # missing_dict = {k:v for k, v in model_state.items() if k not in checkpoint}
    # for key, value in missing_dict.items():
    #     print("Missing {}".format(key))

    # model_state.update(base_dict)
    # net.load_state_dict(model_state)

    net.load_state_dict(checkpoint)
    print("Load weights and bias from /home/zhufl/Temporal-Residual-Motion-Generation/videoPrediction/train/{}".format(model))
    return net

def compare_two_model():

    # bn = pretrained_bninception()
    bn_off = pretrained_bninception_off()

    # weight = bn.conv1_7x7_s2.weight
    # weight_off = bn_off.conv1_7x7_s2.weight

    # print(np.sum(weight.data.numpy() - weight_off.data.numpy())**2)

if __name__ == '__main__':

    # net = pretrained_bninception_off()
    compare_two_model()
