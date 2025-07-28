import numpy as np
from skimage import io
from skimage.segmentation import mark_boundaries

#将处理后的图像和掩码边界标记并保存为文件。如果 item 中包含修补后的图像，它将修补后的图像与原图像拼接在一起并保存。
def save_item_for_vis(item, out_file):# 获取掩码并阈值处理
    mask = item['mask'] > 0.5#将掩码阈值处理，将大于 0.5 的部分设为 True，小于等于 0.5 的部分设为 False。
    if mask.ndim == 3:
        mask = mask[0]
    # 将图像和掩码的通道转换并标记边界，使用 skimage.segmentation.mark_boundaries 在图像上标记掩码边界。
    img = mark_boundaries(np.transpose(item['image'], (1, 2, 0)),
                          mask,
                          color=(1., 0., 0.),
                          outline_color=(1., 1., 1.),
                          mode='thick')
    # 如果有修补后的图像，将其边界标记并与原图像拼接
    if 'inpainted' in item:
        inp_img = mark_boundaries(np.transpose(item['inpainted'], (1, 2, 0)),#将图像的通道从 (C, H, W) 转换为 (H, W, C)。
                                  mask,
                                  color=(1., 0., 0.),
                                  mode='outer')
        img = np.concatenate((img, inp_img), axis=1)#将原图像和修补后的图像沿宽度方向拼接。
    # 将图像像素值限制在 0-255 并保存
    img = np.clip(img * 255, 0, 255).astype('uint8')#将图像像素值限制在 0 到 255 之间，并转换为 uint8 类型
    io.imsave(out_file, img)

#将掩码保存为图像文件。
def save_mask_for_sidebyside(item, out_file):
    mask = item['mask']# > 0.5，获取掩码。
    if mask.ndim == 3:#如果掩码是三维的，取第一维（假设是单通道掩码）。
        mask = mask[0]
    mask = np.clip(mask * 255, 0, 255).astype('uint8')#将掩码像素值限制在 0 到 255 之间，并转换为 uint8 类型。
    io.imsave(out_file, mask)#保存掩码到指定文件。

def save_img_for_sidebyside(item, out_file):#数将原图像保存为文件。
    img = np.transpose(item['image'], (1, 2, 0))#将图像的通道从 (C, H, W) 转换为 (H, W, C)。
    img = np.clip(img * 255, 0, 255).astype('uint8')#将图像像素值限制在 0 到 255 之间，并转换为 uint8 类型。
    io.imsave(out_file, img)#保存图像到指定文件。