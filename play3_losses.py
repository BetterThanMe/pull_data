import os
from shutil import copyfile
dir_src = '/mnt/disk1/minh/pull_data/feat_label'
dir_tar = '/mnt/disk1/minh/pull_data/feat_label2'

num_folds = 6
num_files_each_fole = 7
arr_index = []
for fold_index in range(num_folds):
    for i in range(num_files_each_fole):
        arr_index.append(fold_index+1)

arr_name = {}
for folder in os.listdir(dir_src):
    if folder[0]=='f' and folder!='foa_wts':
        print(folder)
        arr_name[folder] = []
        count = 0
        for file_name in os.listdir(os.path.join(dir_src, folder)):
            if count>len(arr_index)-1:
                break
            if file_name[4]==str(arr_index[count]):
                arr_name[folder].append(file_name)
                count +=1

for folder in arr_name:
    for file_name in arr_name[folder]:
        file_src = os.path.join(dir_src, folder, file_name)
        file_tar = os.path.join(dir_tar, folder, file_name)
        copyfile(file_src, file_tar)

