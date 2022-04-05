import os 
import random

import scipy.io as scio

import tqdm

root = "/media/DATACENTER2/nikolas/dev/data/datasets/ycb_video_dataset/YCB_dataset"
input_file = open(root + "/dataset_config/train_data_list_subset.txt", "r")
file_names = []
file_names_other = []
while True:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]

    if not "syn" in input_line:
        file_names_other.append(input_line)

    if not "syn" in input_line:
        continue

    # Full image path
    file_names.append(
        input_line
    )
input_file.close()

file_names = random.sample(file_names, len(file_names_other))

targets = [3, 10, 13, 15, 16]
output_file = root + "/dataset_config/train_data_list_subset_half.txt"

# new_file_names = []
# for i in tqdm.tqdm(range(len(file_names))):
#     f_in = "{}/{}-meta.mat".format(root, file_names[i])

#     if not os.path.exists(f_in):
#         continue
#     data = scio.loadmat(f_in)

#     obj = list(data['cls_indexes'].flatten().astype(int))
#     if any([t in obj for t in targets]):
#         new_file_names.append(file_names[i])

print("writing out...")
with open(output_file, 'w') as f:
    for l in file_names_other + file_names:
        f.write("{}\n".format(l))