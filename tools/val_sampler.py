import os
import random
import tqdm

val_files_path           = os.path.abspath('.') + '/datasets/Chinese/val/'
list_of_random_val_files = os.path.abspath('.') + '/datasets/Chinese/val_random_files.txt'
val_files      = [val_files_path + image for image in os.listdir(val_files_path)]

random.shuffle(val_files)
val_files_random = val_files[0:5000]

with open(list_of_random_val_files, 'w') as fw:
    for filename in tqdm.tqdm(val_files_random, desc='random_masks'):
        fw.write(filename+'\n')

print('...done')