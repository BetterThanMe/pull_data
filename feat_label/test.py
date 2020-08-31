import numpy as np 

a = np.load('/home/ad/PycharmProjects/Sound_processing/venv/pull_data/feat_label/foa_dev/fold1_room1_mix001_ov1.npy')
print('normal: ',a.shape)

a = np.load('/home/ad/PycharmProjects/Sound_processing/venv/pull_data/feat_label/foa_dev_norm/fold1_room1_mix001_ov1.npy')
print('norm file: ',a.shape)

a = np.load('/home/ad/PycharmProjects/Sound_processing/venv/pull_data/feat_label/foa_dev_label/fold1_room1_mix001_ov1.npy')
print('label file: ',a.shape)

a = np.load('/home/ad/PycharmProjects/Sound_processing/venv/pull_data/feat_label/foa_wts.npy')
print('weight file: ',a.shape)



