import os
import random
import tqdm

val_files_path           = os.path.abspath('.') + '/datasets/Chinese/val/'#验证集图像的目录路径。
list_of_segm_val_files = os.path.abspath('.') + '/datasets/Chinese/eval_random_segm_files.txt'#将保存文件路径的文本文件。
val_files      = [val_files_path + image for image in os.listdir(val_files_path)]#获取验证集目录中的所有图像文件路径。

random.shuffle(val_files)#打乱文件列表的顺序。
val_files_random = val_files[0:4000]#选择前4000个文件。

#打开文本文件，逐行写入每个图像文件的路径。
with open(list_of_segm_val_files, 'w') as fw:
    for filename in tqdm.tqdm(val_files_random, desc='segm masks'):
        fw.write(filename+'\n')

print('...done')

'''
val_files_path 和 list_of_segm_val_files 分别保存验证集图像的目录路径和保存文件路径的文本文件路径。
使用 os.listdir 获取目录中的所有文件名，并生成这些文件的绝对路径列表 val_files。
使用 random.shuffle 随机打乱文件路径列表。
选择打乱后的前4000个文件路径，保存到 val_files_random。
打开 list_of_segm_val_files 文本文件，逐行写入 val_files_random 中的文件路径。
使用 tqdm 模块显示进度条，指示文件路径写入进度。
完成所有操作后，打印 "...done" 消息
'''