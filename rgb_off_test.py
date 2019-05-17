""" Testing the code after training finished
Update 12.21.2018: Be able to do cropping when test for OFF accuracy; Test accuracy on split 1 is 79.00%; Still need to improve to 90.5%;

Update 2019.02.10: 
    1. TODO: save score locally and run in python script/jupyternotebook;
    2. TODO: code for flow branch OFF training;
    3. TODO: code for Feature Generation for UCF101;
"""

from __future__ import division
import os, sys
import numpy as np
import torch
sys.path.append(os.path.join(os.getcwd(), '..'))
from dataset import *
from model_utils import *
from baseModel import *
from transforms import *
import time
from sklearn.metrics import confusion_matrix
import pdb

''' Define config path '''
num_seg = 25
num_batch = 10
weight = 'RGB_OFF_2019-02-13_17-46-14.pth'

''' Init Network '''
BNInceptionNet = selftrained_bninception_off_sobel(batch=num_batch, num_seg=num_seg, model=weight).cuda()
BNInceptionNet.eval()
BNInceptionNet.module.modality_fuse = False

for m in BNInceptionNet.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

# net = Network(224, 224, 3, warmup_length, predict_length).cuda()
# checkpoint = torch.load(weight_model)
# net.load_state_dict(checkpoint['state_dict'])
# print("Loading pre-trained weights for FeaGAN model from {}".format(weight_model))

""" Load train and val data """
normalize = IdentityTransform()

cropping = torchvision.transforms.Compose([
    GroupOverSample(224, 256)
])

input_mean = [104, 117, 123]
input_std = [1]

data_loader = torch.utils.data.DataLoader(
        TSNDataSet("", '../data/ucf101_rgb_val_split_1.txt', num_segments=num_seg,
                   new_length=1 if 'RGB' == "RGB" else 5,
                   modality='RGB',
                   image_tmpl="img_{:05d}.jpg" if 'RGB' in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll= 'BNInception' == 'BNInception'),
                       ToTorchFormatTensor(div = 'BNInception' != 'BNInception'),
                       GroupNormalize(input_mean, input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True
)

accuracy_list = []
l1_distance_list = []
total_num = len(data_loader.dataset)

''' Testing accuracy of oringinal OFF network '''
num_segments = 25

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)

output = []

def eval_video(video_data):
    i, data, label = video_data
    num_crop = 10

    length = 3

    with torch.no_grad():

        data = data.view(-1, length, data.size(2), data.size(3))
        # use partial observed data num_seg * crop : 10 * 10
        data = data[:num_segments * num_crop]
        # print(data.shape)

        input_var = torch.autograd.Variable(data, volatile=True).cuda()

        # rst = BNInceptionNet.RGB_OFF_forward(input_var).data.cpu().numpy().copy()
        '''
        setting 1:
            rst1: 7x7;
            rst2: RGB;
            rst3: 14x14;

        setting 2:
            rst1: 7x7;
            rst2: 28x28;
            rst3: 14x14;
        '''
        if BNInceptionNet.module.modality_fuse is False:
            # rst1, rst2, rst3 = BNInceptionNet.RGB_OFF_forward(input_var)
            rst1, rst2, rst3 = BNInceptionNet(input_var)
            # rst1, rst3 = BNInceptionNet.RGB_OFF_forward(input_var)

            rst1 = rst1.data.cpu().numpy().copy()
            rst2 = rst2.data.cpu().numpy().copy()
            rst3 = rst3.data.cpu().numpy().copy()
            # rst4 = rst4.data.cpu().numpy().copy()

            # rst1 = rst1.reshape(num_segments - 1, 10, 101)
            # rst2 = rst2.reshape(num_segments - 1, 10, 101)
            # rst2 = rst2.reshape(num_segments, 10, 101)
            # rst3 = rst3.reshape(num_segments - 1, 10, 101)
            # rst4 = rst4.reshape(num_segments - 1, 10, 101)

            ''' 
                Merge RGB, OFF 7x7, OFF 14x14:
                According to original implementation, np.mean happens after np.max for crpp dimension ==> [24/25, 101]
                https://github.com/kevin-ssy/Optical-Flow-Guided-Feature/blob/master/tools/ensemble_test.py;
                https://github.com/kevin-ssy/Optical-Flow-Guided-Feature/blob/master/pyActionRecog/utils/video_funcs.py;
                np.max, np.mean result shape is [101], then perform weighted average, then infer the final class label;
        
                Different when do crop mean first ? Should not be;
            '''

            # import pdb;pdb.set_trace()
            # rst = np.mean(rst1, axis=0) + 2 * np.mean(rst2, axis=0)
            rst = np.mean(rst1, axis=0) + 2 * np.mean(rst2, axis=0) + np.mean(rst3, axis=0)

            # ''' Then do np.max on crop dimesnion '''
            rst = rst.reshape(1, 101)

            ''' Use only RGB '''
            ''' Achieve 86.03% / 86.27% '''
            # rst = rst2.mean(axis=0).mean(axis=0).reshape(1, 101)

            ''' User only OFF 7x7 '''
            ''' Achieve 60% '''
            ''' Achieve 79.72% 2019.01.01 '''
            # rst = rst1.mean(axis=0).reshape(1, 101)

            ''' User only OFF 14x14'''
            ''' Achieve 31.53% '''
            # rst = rst3.reshape((10, num_segments - 1, 101)).mean(axis=0).mean(axis=0).reshape(1, 101)
            return i, rst, label[0], rst1, rst2, rst3

        else:
            '''
                When modality_fuse is True, return only one result [num_crop/num_batch, num_class]
            '''
            rst1 = BNInceptionNet.RGB_OFF_forward(input_var)
            rst1 = rst1.data.cpu().numpy().copy()
            rst = np.mean(rst1, axis=0)
            rst = rst.reshape(1, 101)

            return i, rst, label[0]

proc_start_time = time.time()
max_num = -1 if -1 > 0 else len(data_loader.dataset)
# max_num = 10
'''
Notice that below implementation does not utilize the cropping strategy;
Dataloader indeed implement normalize and cropping;
'''

rst1_list = []
rst2_list = []
rst3_list = []
label_list = []

for i, (data, label) in data_gen:

    ''' Must Fully utilize the cropping strategy '''
    data = data.view(25*10, 3, 224, 224) # use 10 cropping strategy
    # data = data[:, 1, :, :, :]

    if i >= max_num:
        break

    rst = eval_video((i, data, label))

    output.append(rst[1:3])
    rst1_list.append(rst[3])
    rst2_list.append(rst[4])
    rst3_list.append(rst[5])
    label_list.append(rst[2])

    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1)))

''' This is wrong, x[0] is already [101], no need to further np.mean '''
''' This can be right, since last implementation reshape rst to be [1, 101], either way works '''
video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
# video_pred = [np.argmax(x[0]) for x in output]

video_labels = [x[1] for x in output]

cf = confusion_matrix(video_labels, video_pred).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
print 'cls_hit:'
print cls_hit
print 'cls_cnt:'
print cls_cnt
cls_acc = cls_hit / cls_cnt
print(cls_acc)
print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

# if 'save_score' is not None:

#     # reorder before saving
#     name_list = [x.strip().split()[0] for x in open('../data/ucf101_rgb_val_split_1.txt')]

#     order_dict = {e:i for i, e in enumerate(sorted(name_list))}

#     reorder_output = [None] * len(output)
#     reorder_label = [None] * len(output)

#     for i in range(len(output)):
#         idx = order_dict[name_list[i]]
#         reorder_output[idx] = output[i]
#         reorder_label[idx] = video_labels[i]

np.savez('rgb_save_score_3', scores1=rst1_list, scores2=rst2_list, scores3=rst3_list, label=label_list)