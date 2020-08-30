#基础4 多种Net On MSCOCO 2017 Dataset Example

#http://images.cocodataset.org/zips/train2017.zip
#http://images.cocodataset.org/annotations/annotations_trainval2017.zip

#http://images.cocodataset.org/zips/val2017.zip
#http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

#http://images.cocodataset.org/zips/test2017.zip
#http://images.cocodataset.org/annotations/image_info_test2017.zip
import os
import time
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉可以调用GPU，不注释时使用CPU
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pycocotools.coco

#pycocotools几个常用API
#构建coco对象 coco = pycocotools.coco.COCO(json_file)
#coco.getImgIds(self, imgIds=[], catIds=[]) 返回满足条件的图像id
#coco.imgs.keys() 数据集中所有样本的id号
#coco.imgToAnns.keys() 数据集中有GT对应的图像样本的id号（用来过滤没有标签的样本）
#coco.getCatIds 返回含有某一类或者几类的类别id号
#coco.loadImgs() 根据id号，导入对应的图像信息
#coco.getAnnIds() 根据id号，获得该图像对应的GT的id号
#coco.loadAnns() 根据 Annotation id号，导入标签信息

path_keypoints = 'annotations/person_keypoints_train2017.json'
path_instances = 'annotations/instances_train2017.json'
path_captions = 'annotations/captions_train2017.json'


COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',*
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

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

#COCO数据集的RLE都是uncompressed RLE格式
#使用下面的代码将这个rle数组表示的分割区域画出来
def readrle(M):
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

# https://tensorflow.google.cn/api_docs/python/tf/keras/applications
# https://keras.io/zh/applications/#applications
std_vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', 
                                              include_top=False, 
                                              pooling=max,
                                              classifier_activation='softmax')
std_vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet', 
                                              include_top=True, 
                                              classifier_activation='softmax')
std_mnet = tf.keras.applications.mobilenet.MobileNet(weights='imagenet',
                                                     dropout=1e-3,
                                                     include_top=False,
                                                     classifier_activation='softmax')
std_mnetv2 = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                            include_top=False,
                                                            classifier_activation='softmax')
std_res50 = tf.keras.applications.resnet50.ResNet50(include_top=True, 
                                                    weights='imagenet')
std_res50v2 = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True,
                                                         weights='imagenet',
                                                         classifier_activation='softmax')
std_res152v2 = tf.keras.applications.resnet_v2.ResNet152V2(include_top=True,
                                                           weights='imagenet',
                                                           classifier_activation='softmax')
std_incv3 = tf.keras.applications.inception_v3.InceptionV3(include_top=True, 
                                              weights='imagenet',
                                              classifier_activation='softmax')

#img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
#x = tf.keras.preprocessing.image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = tf.keras.applications.vgg16.preprocess_input(x)