import numpy as np
from PIL import Image, ImageDraw
import math


def RandomBrush(
        max_tries,#根据 max_tries 的值，随机确定生成形状的次数。
        s,
        min_num_vertex=4,
        max_num_vertex=18,
        mean_angle=2 * math.pi / 5,
        angle_range=2 * math.pi / 15,
        min_width=12,
        max_width=48):
    H, W = s, s
    average_radius = math.sqrt(H * H + W * W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius // 2),
                0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2,
                          v[1] - width // 2,
                          v[0] + width // 2,
                          v[1] + width // 2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask


# 调用 RandomBrush 函数生成随机绘制的形状，结合原始掩码，形成最终的掩码。
def RandomMask(s, hole_range=[0, 1]):  # hole_range 是一个列表，表示生成掩码时允许的孔洞比例范围。
    coef = min(hole_range[0] + hole_range[1], 1.0)  # coef 计算孔洞填充的系数，限制在 0 到 1 之间
    while True:
        mask = np.ones((s, s), np.uint8)

        # 使用 Fill 和 MultiFill 函数填充随机大小和位置的矩形区域，生成初步的掩码。
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0

        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)

        MultiFill(int(10 * coef), s // 2)
        MultiFill(int(5 * coef), s)
        mask = 1 - np.logical_and(mask, 1 - RandomBrush(int(20 * coef), s))
        hole_ratio = np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)


# 接受两个参数：s 表示掩码的大小（宽度和高度），默认为 256；hole_range 是一个列表，表示生成掩码时允许的孔洞比例范围，默认为 [0.1, 1]。
# 调用 RandomMask(s, hole_range) 函数生成随机掩码，并返回生成的掩码。
def generate_random_mask(s=256, hole_range=[0.1, 1]):
    return RandomMask(s, hole_range)
