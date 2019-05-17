"""
    1. Re-train OFF network for better performance on UCF101;
    2. Freeze TSN network parameter;
    3. fine-tune the last fc layer for 7x7 and 14x14;
    4. Or fine-tune whole motion branch;

Notice: In OFF paper, OFF is trained also with 7/8 input sequence;
Notice: In OFF paper, OFF is trained with 128 as batch size;
Notice: Do testing on val set with cropping while training is hard;
        Since batch size is different for cropping and network is not eary to adjust to evaluation mode;
        What I can do is do testing on both train set and val set after training finished;
Notice: The sampling strategy is also different;
"""

from __future__ import division
import torch
import os, sys
import numpy as np
from torch.nn.utils import clip_grad_norm
sys.path.append(os.path.join(os.getcwd(), '..'))

from baseModel import *
from transforms import *
from model_utils import *
from basic_ops import *
from dataset_off import *
import pdb, datetime

num_seg = 7
num_batch = 42 # Should be 128, if result not good, then try to run on server;

# Init feature extractor network, output [batch, 192, 56, 56] tensor
net = fine_tune_bninception_off(batch=num_batch, num_seg=num_seg).cuda()

# Freeze most parameter
# any(substring in string for substring in substring_list)
for name, param in net.named_parameters():
    if any(sub_name in name for sub_name in ['fc_action_motion_14', 'fc_action_motion_28', 'fc_action_motion', 'motion']):
    # if any(sub_name in name for sub_name in ['fc_action_motion', 'motion']):
        param.requires_grad = True
        print("fine tuning para name {}, shape {}".format(name,param.data.shape))
    else:
        param.requires_grad = False
        # print("Do not fine tuning para name {}, shape {}".format(name,param.data.shape))

''''
    Freeze all parameters of BN layer
    This will make sure that original TSN feature doesn not change at all; Based on this, learning motion branch is useful;
    In tensorflow code, need to pay attention this;
'''
for m in net.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

# idx = 0
# for name, param in net.named_parameters():
#     idx += 1
#     if param.requires_grad:
#         print(idx, name, param.shape)

# Define loss func and optimizer
criterion = nn.CrossEntropyLoss()
param = filter(lambda p: p.requires_grad, net.parameters())

'''
    policy learning strategy:
        For 0-10 epoch: lr=1e-3;
        For 10-20 epoch: lr=1e-4;
        For 20-40 epoch: lr=5e-5;
        For 40-60 epoch: lr=1e-6;
'''
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(param, lr = 0.001, betas= (0.9, 0.99), weight_decay=0.0005)

# Init dataloader of ucf101
normalize = IdentityTransform()

train_loader = torch.utils.data.DataLoader(
    TSNDataSet("", '../data/ucf101_rgb_train_split_1.txt' , num_segments=num_seg,
                new_length=1,
                modality='RGB',
                image_tmpl="img_{:05d}.jpg" if 'RGB' in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.Compose([GroupMultiScaleCrop([224, 224], [1, .875, .75, .66]),
                                                GroupRandomHorizontalFlip(is_flow=False)]),
                    Expand(roll='BNInception' == 'BNInception'),
                    ToTorchFormatTensor_expand(div='BNInception' != 'BNInception'),
                    normalize,
                ])),
    batch_size=num_batch, shuffle=True,
    num_workers=1, pin_memory=True, drop_last=True
)
print(len(train_loader))

# val_loader = torch.utils.data.DataLoader(
#     TSNDataSet("", '../data/ucf101_rgb_val_split_1.txt', num_segments=num_seg,
#                 new_length=1,
#                 modality='RGB',
#                 image_tmpl="img_{:05d}.jpg" if "RGB" in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
#                 random_shift=False,
#                 transform=torchvision.transforms.Compose([
#                 #    GroupScale(int(scale_size)),
#                 #    GroupCenterCrop(crop_size),
#                     Stack(roll= 'BNInception' == 'BNInception'),
#                     ToTorchFormatTensor(div= 'BNInception' != 'BNInception'),
#                     normalize,
#                 ])),
#     batch_size=num_batch, shuffle=False,
#     num_workers=1, pin_memory=True,  drop_last=True)
# print(len(val_loader))


# for epoch in range(20):
#     for idx, (input, target) in enumerate(train_loader):
#         import pdb;pdb.set_trace()
#         print("epoch {}, batch_id {}".format(epoch, idx))
#         input = input.permute(1,0,2,3,4).cuda()

#         '''
#             No grad operation save memory
#             Get intermediate feature from BNInception Network
#             Return a list of sequence [batch, channel, height, width]
#             Compute feature diff offline
#         '''

#         feature_diff_list = []
#         feature_list = []
#         with torch.no_grad():
#             for i in input:
#                 feature_list.append(featureNet.extract_feature(i))
#             # print("feature length is {}".format(len(feature_list)))

#             for i, j in zip(feature_list[:-1], feature_list[1:]):
#                 feature_diff_list.append(j-i)
#             # print("feature diff length is {}".format(len(feature_diff_list)))

#         ''' Start training mse process '''
#         net.train(feature_list, feature_diff_list)

#         ''' Start training gan process '''

#     ''' save model after each epoch '''
#     is_best = None
#     save_checkpoint({
#         'epoch': epoch + 1,
#         'arch': 'BNInception',
#         'state_dict': net.state_dict(),
#         'optimizer' : net.optimizer_g.state_dict(),
#     }, is_best)

''' fine tune motion-branch/fc layer of OFF network '''
for epoch in range(20):

    ''' Notice: output indice to see if different sampling strategy output different offsets '''
    for idx, (input, target) in enumerate(train_loader):

        # import pdb;pdb.set_trace()
        # early stop for testing
        # if idx > 50:
        #     break

        # merge batch dimension with frame number
        data = input.view(-1, 3, input.size(-2), input.size(-1))
        input_var = torch.autograd.Variable(data).cuda()

        optimizer.zero_grad()
        '''
            rst1 : 7x7
            rst2 : 28x28
            rst3 : 14x14
        '''
        rst1, rst2, rst3 = net.RGB_OFF_forward(input_var)

        rst1 = rst1.view(num_batch * (num_seg - 1), -1)
        rst2 = rst1.view(num_batch *  (num_seg -1), -1)
        rst3 = rst1.view(num_batch * (num_seg - 1), -1)

        ''' repeat target tensor to [batch * frame] '''
        target = target.unsqueeze(1).repeat(1, num_seg-1).view(-1).cuda()

        # output = loss(input, target)
        # values, indices = torch.max(tensor, 0)
        ''' 7x7 loss '''
        loss1 = criterion(rst1, target)

        ''' 14x14 loss '''
        loss2 = criterion(rst3, target)

        ''' 28x28 loss '''
        loss3 = criterion(rst2, target)

        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        # loss2.backward()
        loss3.backward()

        ''' gradient clip '''
        clip_grad_norm(param, max_norm=20)

        optimizer.step()

        print("Epoch {}, data batch {}, 7x7 loss is {}, 14x14 loss is {}, 28x28 loss is {}".format(epoch, idx, loss1.data[0].cpu(), loss2.data[0].cpu(), loss3.data[0].cpu()))
        # print("Epoch {}, data batch {}, 7x7 loss is {}, 14x14 loss is {}, 28x28 loss is".format(epoch, idx, loss1.data[0].cpu(), loss2.data[0].cpu()))
        # print("Epoch {}, data batch {}, 7x7 loss is {}".format(epoch, idx, loss1.data[0].cpu()))

state_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.pth'
torch.save(net.state_dict(), state_name)
