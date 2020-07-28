#MSCOCO 2017 Dataset example

#http://images.cocodataset.org/zips/train2017.zip
#http://images.cocodataset.org/annotations/annotations_trainval2017.zip

#http://images.cocodataset.org/zips/val2017.zip
#http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

#http://images.cocodataset.org/zips/test2017.zip
#http://images.cocodataset.org/annotations/image_info_test2017.zip
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用cpu，不注释就是使用gpu
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#annotations字段是包含多个annotation实例的一个数组.
#segmentation格式取决于这个实例是一个单个的对象（即iscrowd=0，使用polygons格式）
#还是一组对象（即iscrowd=1，使用RLE格式）

#一段解析polygon格式的segmentation并且显示多边形的示例代码
fig, ax = plt.subplots()
polygons = []
num_sides = 100
gemfield_polygons = []
gemfield_polygon = gemfield_polygons[0]
max_value = max(gemfield_polygon) * 1.3
gemfield_polygon = [i * 1.0 / max_value for i in gemfield_polygon]
poly = np.array(gemfield_polygon).reshape((int(len(gemfield_polygon) / 2), 2))
polygons.append(Polygon(poly,True))
p = matplotlib.collections.PatchCollection(polygons, cmap=matplotlib.cm.jet, alpha=0.4)
colors = 100 * np.random.rand(1)
p.set_array(np.array(colors))

ax.add_collection(p)
plt.show()

#COCO数据集的RLE都是uncompressed RLE格式.使用下面的代码将这个rle数组表示的分割区域画出来
rle = []
assert sum(rle) == 240 * 320
M = np.zeros(240 * 320)
N = len(rle)
n = 0
val = 1
for pos in range(N):
    val = not val
    for c in range(rle[pos]):
        M[n] = val
        n += 1

GEMFIELD = M.reshape(([240, 320]), order='F')
plt.imshow(GEMFIELD)
plt.show()

