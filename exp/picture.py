"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# 加载图像
img = cv2.imread('/home/code/unsupervised-light-enhance-ICLR2025-main/results/middle_org/I/83.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 设置局部放大区域（左上角点、宽高）
x, y, w, h = 1000, 600, 200, 120  # 根据图像选择

# 创建图像窗口
fig, ax = plt.subplots()

# 显示原图
ax.imshow(img)

# 在主图中加一个框（表示要放大的区域）
rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lightblue', facecolor='none')
ax.add_patch(rect)

# 添加一个 inset axes 显示放大区域
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes

# axins = zoomed_inset_axes(ax, zoom=3, loc='upper right')  # 放大倍率、位置
axins = inset_axes(ax, width="200%", height="200%", bbox_to_anchor=(1.0, 1.0, 0.1, 0.1),
                   bbox_transform=ax.transAxes, borderpad=1)
axins.imshow(img)
axins.set_xlim(x, x + w)
axins.set_ylim(y + h, y)  # 注意y轴方向相反

# 去掉坐标轴
axins.axis('off')

# 在放大图中加框（用于标出放大区域）
# rect_zoom = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lightblue', facecolor='none')
# axins.add_patch(rect_zoom)

# 画连接线
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="lightblue")

plt.axis('off')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cv2

# -------- 参数设置 --------
image_path = '/home/code/unsupervised-light-enhance-ICLR2025-main/results/middle_org/I/83.jpg'
zoom_color = 'lightcoral'
zoom_regions = [(130, 150, 20, 20), (130, 250, 20, 20)]  # x, y, w, h
zoom_factor = 6  # 放大倍数

# -------- 加载图像 --------
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, _ = img.shape

# -------- 创建主图 --------
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(img)
ax.axis('off')

# -------- 绘制主图区域框 + 放大图 --------
for i, (x, y, w, h) in enumerate(zoom_regions):
    # 在主图中画框
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=zoom_color, facecolor='none')
    ax.add_patch(rect)
    
    # 创建放大图（使用inset_axes）
    axins = inset_axes(ax, width="15%", height="15%", 
                       loc='center right', 
                       bbox_to_anchor=(0.98, 0.65 - i*0.35, 1, 1),  # 垂直排列位置
                       bbox_transform=ax.transAxes)
    
    # 显示放大区域
    axins.imshow(img)
    axins.set_xlim(x, x + w)
    axins.set_ylim(y + h, y)  # 注意y轴方向
    
    # 在放大图上添加红色边框
    rect_zoom = patches.Rectangle((x, y), w, h, linewidth=2, 
                                 edgecolor=zoom_color, facecolor='none')
    axins.add_patch(rect_zoom)
    
    # 添加放大指示线
    ax.annotate('', xy=(x+w, y+h/2), xytext=(1.02, 0.65 - i*0.35),
                arrowprops=dict(arrowstyle="->", color=zoom_color, linewidth=1.5),
                xycoords='data', textcoords=ax.transAxes)
    
    # 添加放大标记
    axins.text(0.05, 0.95, f'{zoom_factor}X', 
              transform=axins.transAxes, 
              color='white', backgroundcolor=zoom_color,
              fontsize=10, fontweight='bold')

# 调整布局，确保放大图靠近主图
plt.subplots_adjust(right=0.85)
plt.tight_layout()
plt.show()

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
# img = cv2.imread('/home/code/unsupervised-light-enhance-ICLR2025-main/results/middle_org/I/83.jpg')
img = cv2.imread('2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 定义两个框的位置：[x1, y1, x2, y2]
boxes = [
    [300, 0, 300, 264],  # 90, 53
    [900, 750, 300, 264]
]

colores = [
    (255, 0, 0),
    (0, 255, 0)
]

# 放大倍数
scale = 2
zoomed_patches = []

# 原图复制，用于绘制矩形
img_draw = img.copy()

for i in range(0, len(boxes)):
    box = boxes[i]
    x1, y1, W, H = box
    x2 = x1 + W
    y2 = y1 + H
    # 画框（线宽固定为1-2）
    # if i == 0:
    cv2.rectangle(img_draw, (x1, y1), (x2, y2), color=colores[i], thickness=2)
    
    # 裁剪原始区域（不带框）
    patch = img[y1:y2, x1:x2]
    # 放大
    patch_zoomed = cv2.resize(patch, (patch.shape[1]*scale, patch.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    # 添加边框（统一固定宽度）
    patch_zoomed = cv2.copyMakeBorder(patch_zoomed, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=colores[i])
    zoomed_patches.append(patch_zoomed)

# 垂直拼接放大图
zoomed_column = np.vstack(zoomed_patches)

# 补齐原图高度以匹配放大图
height_diff = zoomed_column.shape[0] - img_draw.shape[0]
if height_diff > 0:
    img_padded = cv2.copyMakeBorder(img_draw, 0, height_diff, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
else:
    img_padded = img_draw
    zoomed_column = cv2.copyMakeBorder(zoomed_column, 0, -height_diff, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

# 水平拼接
result = np.hstack((img_padded, zoomed_column))

# 显示结果
plt.figure(figsize=(12, 6))
plt.imshow(result)
plt.axis('off')
plt.tight_layout()
plt.show()


