import os
import random
import tqdm

val_files_path           = os.path.abspath('.') + '/datasets/Chinese/val/'#验证集图像的目录路径。
list_of_random_val_files = os.path.abspath('.') + '/datasets/Chinese/eval_random_files.txt'#将保存随机选择文件路径的文本文件路径。
val_files      = [val_files_path + image for image in os.listdir(val_files_path)]#使用 os.listdir 获取目录中的所有文件名。将文件名与目录路径拼接，生成完整的文件路径列表 val_files。

random.shuffle(val_files)#使用 random.shuffle 随机打乱文件列表的顺序。
val_files_random = val_files[0:20000]#选择打乱后的前 20,000 个文件路径，保存到 val_files_random。

with open(list_of_random_val_files, 'w') as fw:#打开 list_of_random_val_files 文本文件。使用 tqdm 模块显示进度条，逐行写入 val_files_random 中的文件路径。
    for filename in tqdm.tqdm(val_files_random, desc='random_masks'):
        fw.write(filename+'\n')

print('...done')

'''
定义验证集图像的目录路径和保存随机选择文件路径的文本文件路径。
使用 os.listdir 获取目录中的所有文件名，并生成这些文件的绝对路径列表 val_files。
使用 random.shuffle 随机打乱文件路径列表。
选择打乱后的前 30,000 个文件路径，保存到 val_files_random。
打开 list_of_random_val_files 文本文件，逐行写入 val_files_random 中的文件路径，并使用 tqdm 模块显示进度条，指示文件路径写入进度。
完成所有操作后，打印 "...done" 消息。
'''