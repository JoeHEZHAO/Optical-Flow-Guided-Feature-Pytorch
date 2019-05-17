""" Testing the code after training finished
Update 12.21.2018: Be able to do cropping when test for OFF accuracy; Test accuracy on split 1 is 79.00%; Still need to improve to 90.5%;

Update 2019.02.10: 
    1. TODO: save score locally and run in python script/jupyternotebook;[done]
    2. TODO: code for flow branch OFF training;[done]
    3. TODO: code for Feature Generation for UCF101;[done]
    4. TODO: Adjust Flow_OFF to dataset_off.py sampling strategy;[done]
    5. TODO: Equip Residual Motion Generator for the mechanism;[done]
    6. TODO: Save score into local; So that testing can be faster; [done]
"""
from __future__ import division
import os, sys
import numpy as np
import torch
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..'))

from dataset import *
from transforms import *
sys.path.append(os.path.join(cur_path, '../../tsn-pytorch'))
from models_off import TSN

import time
from sklearn.metrics import confusion_matrix
import pdb

''' Define config/path/setting '''
num_seg = 25
num_batch = 10
num_class = 101
modality = 'Flow'
crop_fusion_type = 'avg'
arch = 'BNInception'
flow_prefix = 'flow_'

''' Init Network '''
net = TSN(num_class, num_segments=num_seg, modality=modality, batch_size=num_batch, length=num_seg,
            base_model=arch,
            consensus_type=crop_fusion_type,
            dropout=0.7)
net.base_model.modality_fuse = False
net.eval()
print(net)

net = torch.nn.DataParallel(net, device_ids=[0]).cuda()

''' Load pretrained weights '''
model_name = 'Flow_OFF_2019-02-14_19-13-18.pth'
checkpoint = torch.load(os.path.join(cur_path, '../train', model_name))
checkpoint2 = {}
for k,v in checkpoint.items():
    if 'last_linear' not in k:
        k = k.replace('module.', 'module.base_model.')
    else:
        k = k.replace('last_linear', 'new_fc')
        print(k)
    checkpoint2[k] = v
print("Number of parameters recovered from modeo {} is {}".format(model_name, len(checkpoint)))
# import pdb;pdb.set_trace()

# ''' Load fc-action from ucf101_flow.pth '''
# checkpoint = torch.load(os.path.join(cur_path, '../../tsn-pytorch', 'ucf101_flow.pth'))
# checkpoint2['module.new_fc.weight'] = checkpoint['base_model.fc-action.1.weight']
# checkpoint2['module.new_fc.bias'] = checkpoint['base_model.fc-action.1.bias']

net.load_state_dict(checkpoint2)

cropping = torchvision.transforms.Compose([
    GroupOverSample(net.module.input_size, net.module.scale_size)
])

data_loader = torch.utils.data.DataLoader(
        TSNDataSet("", '../data/ucf101_flow_val_split_1.txt', num_segments=num_seg,
                   new_length=1 if modality == "RGB" else 5,
                   modality=modality,
                   image_tmpl="img_{:05d}.jpg" if modality in ['RGB', 'RGBDiff'] else flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=arch == 'BNInception'),
                       ToTorchFormatTensor(div=arch != 'BNInception'),
                       GroupNormalize(net.module.input_mean, net.module.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True
)

accuracy_list = []
l1_distance_list = []
total_num = len(data_loader.dataset)

''' Testing accuracy with predicted feature '''
# for idx, (input, target) in enumerate(data_loader):

#     # permute batch/crop and time axis
#     input = input.view(25, -1, 3, input.size(2), input.size(3)).cuda()

#     feature_diff_list = []
#     feature_list = []

#     '''  compute intermediate output of feature extraction network'''
#     with torch.no_grad():
#         for i in input:
#             feature_list.append(BNInceptionNet.extract_feature(i))

#         for i, j in zip(feature_list[:-1], feature_list[1:]):
#             feature_diff_list.append(j-i)

#         ''' predicted feature for next x time steps '''
#         pred_feature, pred_diff = net(feature_list[:warmup_length], feature_diff_list[:(warmup_length-1)])


#         """ l1 distance between ground truth and prediction """
#         l1_distance = 0.0
#         for gt, pred in zip(feature_list[warmup_length:warmup_length+predict_length], pred_feature):
#             l1_distance += np.mean((pred.data[0].cpu().numpy() - gt.data[0].cpu().numpy())**2)
#         l1_distance_list.append(l1_distance)
#         print("l1 norm distance for data batch {} is {}".format(idx, (l1_distance / predict_length)))

#         """ generate label based on both warmup frames and generated """
#         """ 10 crop average """
#         """ temporal pooling """
#         logits = []
#         for i in feature_list[:warmup_length]:
#             logits.append(BNInceptionNet.intermediate_cls(i))

#             # value, indice = torch.max(logits, 1)
#             # warmup_label_list.append(indice.data.cpu().numpy())
#         # pred_label_list = []

#         for i in pred_feature:
#             logits.append(BNInceptionNet.intermediate_cls(i))
#             # value, indice = torch.max(logits, 1)
#             # pred_label_list.append(indice.data.cpu().numpy())
#         # final_label_list = warmup_label_list + pred_label_list

#         # combine crop and time-segment
#         logits = torch.stack(logits, dim=1).data.cpu().numpy() # [crop, time-seg, num_class], [10, 20, 101]
#         logits = np.mean(logits, axis=0).reshape(warmup_length+predict_length, 1, 101) # average on crop axis
#         logits = np.argmax(np.mean(logits, axis=0), axis=1) # average on time-seg axis

#         # value, indice = torch.max(logits, 1) # temporal average pooling
#         accuracy = np.sum(target.data.cpu().numpy() == logits)
#         accuracy_list.append(accuracy)
#         print("Accuracy for batch {} is {} with prediction {} and label {}".format(idx, np.sum(target.data.cpu().numpy() == logits), \
#         target.data.cpu().numpy(), logits))
#         # print("Ground truth label is {}, while output of predicted feature is {}".format(target))
#         del feature_diff_list, feature_list, pred_feature, pred_diff, logits, l1_distance, input

# print("Final accuracy for whole validation dataset is {}".format(sum(accuracy_list) / len(accuracy_list)))
# print("Final average l1 distance for whole validation dataset is {}".format(sum(l1_distance_list) / len(l1_distance_list) / (predict_length)))

''' Testing accuracy with 10 crop and predicted feature '''
''' Crop index can be viewed as batch '''
''' Shoule I use code fore directly ? '''
# def eval_video(video_data):
#     i, data, label = video_data
#     num_crop = 10
#     length = 3

#     ''' for warmup 10 frames '''
#     with torch.no_grad():

#         data = data.view(-1, length, data.size(2), data.size(3))

#         input_var = torch.autograd.Variable(data).cuda()

#         rst = BNInceptionNet(input_var).data.cpu().numpy().copy()

#         return i, rst.reshape((10, 25, 101)).mean(axis=0).reshape(
#             (25, 1, 101)
#         ), label[0]

# data_gen = enumerate(data_loader)

# total_num = len(data_loader.dataset)

# for i, (data, label) in data_gen:

#     data = data.view(10, -1, 3, data.size(2), data.size(3)).cuda() # [crop, segment, channel, height, width]

#     # Arrange data to [crop1, seg, channel, height, width], [crop2, seg, channel, height, width] ...
#     # Extract feature  as [seg, crop1, feature], [seg, crop2, feature], [seg, crop3, feature] ...

#     for k in data:
#         feature_list = []
#         feature_diff_list = []

#         feature_list.append(BNInceptionNet.extract_feature(k))
#         pdb.set_trace()

#         for i, j in zip(feature_list[:-1], feature_list[1:]):
#             feature_diff_list.append(j-i)

#         ''' predicted feature for next x time steps '''
#         pred_feature, pred_diff = net(feature_list[:warmup_length], feature_diff_list[:(warmup_length-1)])
#         pdb.set_trace()


''' Testing accuracy from original implementation of tsn-pytorch network '''
# num_segments = 25

# data_gen = enumerate(data_loader)

# total_num = len(data_loader.dataset)

# output = []

# def eval_video(video_data):
#     i, data, label = video_data
#     num_crop = 10

#     length = 3

#     with torch.no_grad():

#         data = data.view(-1, length, data.size(2), data.size(3))

#         # use partial observed data num_seg * crop : 10 * 10
#         data = data[:num_segments * num_crop]

#         input_var = torch.autograd.Variable(data).cuda()

#         # rst = BNInceptionNet(input_var).data.cpu().numpy().copy()
#         rst = BNInceptionNet(input_var).data.cpu().numpy().copy()

#         return i, rst.reshape((10, num_segments, 101)).mean(axis=0).reshape(
#             (num_segments, 1, 101)
#         ), label[0]

# proc_start_time = time.time()
# max_num = -1 if -1 > 0 else len(data_loader.dataset)
# max_num = 100

# for i, (data, label) in data_gen:

#     if i >= max_num:
#         break

#     rst = eval_video((i, data, label))
#     output.append(rst[1:])

#     cnt_time = time.time() - proc_start_time
#     print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
#                                                                     total_num,
#                                                                     float(cnt_time) / (i+1)))

# video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

# video_labels = [x[1] for x in output]

# pdb.set_trace()

# cf = confusion_matrix(video_labels, video_pred).astype(float)

# cls_cnt = cf.sum(axis=1)
# cls_hit = np.diag(cf)
# print 'cls_hit:'
# print cls_hit
# print 'cls_cnt:'
# print cls_cnt
# cls_acc = cls_hit / cls_cnt

# print(cls_acc)

# print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

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
#     np.savez('save_score', scores=reorder_output, labels=reorder_label)


''' Testing accuracy with original feature without cropping '''
''' Notice that still need to sample 25 frames in dataloader, since it is averagely sampled '''
''' Then only use half of them to do test '''
# for idx, (input, target) in enumerate(val_loader):
#     input = input.permute(1,0,2,3,4).cuda()

#     logits = 0.0
#     with torch.no_grad():
#         for i in input[:20]:
#             logits += BNInceptionNet(i)

#     value, indice = torch.max(logits, 1)
#     accuracy = np.sum(target.data.cpu().numpy() == indice.data.cpu().numpy())
#     accuracy_list.append(accuracy)
#     print("Accuracy for batch {}".format((accuracy / 16)))
# print("Final accuracy for whole validation dataset is {}".format(sum(accuracy_list) / len(accuracy_list) / 16))

''' Testing accuracy of oringinal OFF network '''
num_segments = 25

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)

output = []

# score list for 3 output
def eval_video(video_data):
    i, data, label = video_data
    num_crop = 10

    length = 10

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

        if net.module.base_model.modality_fuse is False:
            # rst1, rst2, rst3 = BNInceptionNet.RGB_OFF_forward(input_var)
            rst1, rst2, rst3 = net(input_var)
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

            ''' Compound Method '''
            # rst = np.mean(rst1, axis=0) + 2 * np.mean(rst2, axis=0) + np.mean(rst3, axis=0)

            ''' Use Flow only '''
            ''' Achieve  '''
            # rst = rst2.mean(axis=0)

            ''' User only OFF 7x7 '''
            ''' 82.81% '''
            rst = rst1.mean(axis=0).reshape(1, 101)

            ''' User only OFF 14x14'''
            ''' Achieve 31.53% '''
            # rst = rst3.reshape((10, num_segments - 1, 101)).mean(axis=0).mean(axis=0).reshape(1, 101)

            rst = rst.reshape(1, 101)

            return i, rst, label[0], rst1, rst2, rst3

        else:
            '''
                When modality_fuse is True, return only one result [num_crop/num_batch, num_class]
            '''
            rst1 = net(input_var)
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
    data = data.view(25*10, 10, 224, 224) # use 10 cropping strategy

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

np.savez('flow_save_score', scores1=rst1_list, scores2=rst2_list, scores3=rst3_list, label=label_list)

# if 'save_score' is not None:

#     # reorder before saving
#     name_list = [x.strip().split()[0] for x in open('../data/ucf101_rgb_val_split_1.txt')]

#     order_dict = {e:i for i, e in enumerate(sorted(name_list))}

#     reorder_output = [[None] * 3] * len(rst7x7_list)
#     reorder_label = [None] * len(rst7x7_list)
#     vid_name = [None] * len(rst7x7_list)

#     for i in range(len(output)):
#         idx = order_dict[name_list[i]]
        
#         reorder_output[idx][0] = rst7x7_list[i]
#         reorder_output[idx][1] = rst_org_list[i]
#         reorder_output[idx][2] = rst14x14_list[i]
#         reorder_label[idx] = lable_list[i]

#     np.savez('flow_save_score', scores=reorder_output, label=reorder_label)