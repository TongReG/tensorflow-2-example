#基础4 多种Net On MSCOCO 2017 Dataset Example
#你需要先下载COCO 2017数据集并放入本仓库根目录下，或手动更改path
#然后手动运行coco_init.py以生成TFRECORD文件集，大小约15GB
#本文件将读取TFRECORD，以加载数据
import os
import sys
import platform
import time
import random
from multiprocessing import Process, Queue, cpu_count, freeze_support
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉可以调用GPU，不注释时使用CPU
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pycocotools.coco
import cv2

if platform.system() == 'Linux':
    import resource
    def limit_memory(maxsize):
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

#object instances（目标实例）object keypoints（目标上的关键点）image captions（看图说话）
#pycocotools几个常用API
#构建coco对象: coco = pycocotools.coco.COCO(json_file)
#coco.getImgIds(self, imgIds=[], catIds=[]) 返回满足条件的图像id
#coco.imgs.keys() 数据集中所有样本的id号
#coco.imgToAnns.keys() 数据集中有Ground Truth的图像样本的id号（用来过滤无标签样本）
#coco.getCatIds 返回含有某一类或者几类的类别id号
#coco.loadImgs() 根据id号，导入对应的图像信息
#coco.getAnnIds() 根据id号，获得该图像对应的Ground Truth id
#coco.loadAnns() 根据Annotation id导入标签信息
path_keypoints = 'annotations/person_keypoints_train2017.json'
path_instances = 'annotations/instances_train2017.json'
path_captions = 'annotations/captions_train2017.json'
#COCO训练集的图像数据保存的目录
img_path = 'train2017/'
catas_dict = {}
imgrcg_box = {}
cores = int(cpu_count() / 2)   #定义用到的CPU处理的核心数
if cores >= 6:
    cores = 6
max_num = 2048   #每个TFRECORD文件包含的最多的图像数


#解析TFRECORD文件 https://juejin.im/post/6844903760699850765
def _parse_function(example_proto):
    features = {"image": tf.io.FixedLenFeature([], tf.string, default_value=""),
                "height": tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),
                "width": tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),
                "channels": tf.io.FixedLenFeature([1], tf.int64, default_value=[3]),
                "colorspace": tf.io.FixedLenFeature([], tf.string, default_value=""),
                "img_format": tf.io.FixedLenFeature([], tf.string, default_value=""),
                "label": tf.io.VarLenFeature(tf.int64),
                "bbox_xmin": tf.io.VarLenFeature(tf.int64),
                "bbox_xmax": tf.io.VarLenFeature(tf.int64),
                "bbox_ymin": tf.io.VarLenFeature(tf.int64),
                "bbox_ymax": tf.io.VarLenFeature(tf.int64),
                "filename": tf.io.FixedLenFeature([], tf.string, default_value="")
               }
    # https://tensorflow.google.cn/api_docs/python/tf/io/parse_single_example
    parsed_features = tf.io.parse_single_example(example_proto, features)
    label = tf.expand_dims(parsed_features["label"].values, 0)
    label = tf.cast(label, tf.int64)
    height = parsed_features["height"]
    width = parsed_features["width"]
    channels = parsed_features["channels"]

    image_raw = tf.image.decode_jpeg(parsed_features["image"], channels=3)
    image_decoded = tf.image.convert_image_dtype(image_raw, tf.float32)

    image_standard = tf.image.per_image_standardization(image_decoded)
    image_train = tf.transpose(image_standard, perm=[2, 0, 1])

    xmin = tf.expand_dims(parsed_features["bbox_xmin"].values, 0)
    xmax = tf.expand_dims(parsed_features["bbox_xmax"].values, 0)
    ymin = tf.expand_dims(parsed_features["bbox_ymin"].values, 0)
    ymax = tf.expand_dims(parsed_features["bbox_ymax"].values, 0)

    xmin = tf.cast(xmin, tf.int64)
    xmax = tf.cast(xmax, tf.int64)
    ymin = tf.cast(ymin, tf.int64)
    ymax = tf.cast(ymax, tf.int64)

    bbox = tf.concat(axis=0, values=[xmin, xmax, ymin, ymax, label])
    bbox = tf.transpose(bbox, [1, 0])
    
    return bbox, image_train, image_decoded

def NNAPI(str):
    # https://tensorflow.google.cn/api_docs/python/tf/keras/applications
    # https://keras.io/zh/applications/#applications
    if str == "vgg16":
        std_vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', 
                                                  include_top=False, 
                                                  pooling=max,
                                                  classifier_activation='softmax')
        return std_vgg16
    elif str == "vgg19":
        std_vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet', 
                                                  include_top=True, 
                                                  classifier_activation='softmax')
        return std_vgg19
    elif str == "mobilenet":
        std_mnet = tf.keras.applications.mobilenet.MobileNet(weights='imagenet',
                                                         dropout=1e-3,
                                                         include_top=False,
                                                         classifier_activation='softmax')
        return std_mnet
    elif str == "mobilenetv2":
        std_mnetv2 = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                include_top=False,
                                                                classifier_activation='softmax')
        return std_mnetv2
    elif str == "ResNet50":
        std_res50 = tf.keras.applications.resnet50.ResNet50(include_top=True, 
                                                        weights='imagenet')
        return std_res50
    elif str == "ResNet50v2":
        std_res50v2 = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True,
                                                             weights='imagenet',
                                                             classifier_activation='softmax')
        return std_res50v2
    elif str == "ResNet152v2":
        std_res152v2 = tf.keras.applications.resnet_v2.ResNet152V2(include_top=True,
                                                               weights='imagenet',
                                                               classifier_activation='softmax')
        return std_res152v2
    elif str == "Inceptionv3":
        std_incv3 = tf.keras.applications.inception_v3.InceptionV3(include_top=True, 
                                                               weights='imagenet',
                                                               classifier_activation='softmax')
        return std_incv3
    elif str == "Inceptionv4":
        std_incv4 = tf.keras.applications.InceptionResNetV2(include_top=True, weights='imagenet', classifier_activation='softmax')
        return std_incv4
    else: 
        print('\nNo that Neural Network NAME SUPPORTED, WILL EXIT.','\n')
        sys.exit(1)

if __name__ == '__main__':
    freeze_support()
    coco_train = pycocotools.coco.COCO(path_instances)
    COCO_CLASSES = coco_train.dataset['categories']
    train_ids = list(coco_train.imgToAnns.keys())
    print(COCO_CLASSES)
    #if len(self.ids) == 0: # 如果没有标签或者不需要GT，则直接使用image
    #    train_ids = list(coco_train.imgs.keys())

    # display COCO categories and supercategories
    catas = coco_train.loadCats(coco_train.getCatIds())
    for cata in catas:
        catas_dict[cata['id']] = cata['name']
 
    #获取COCO数据集中所有图像的ID,构建训练集文件列表，里面的每个元素是路径名+图片文件名
    trainimg_ids = coco_train.getImgIds()
    print('trainimg_NUMBERS =>',len(trainimg_ids))
    train_images_filenames = os.listdir(img_path)
    #查找训练集的图片是否都有对应的ID，并保存到一个列表中
    trainimg_path = []
    i = 1
    total = len(train_images_filenames)
    #for image_names in train_images_filenames:
    #    if int(image_names[:-4]) in trainimg_ids:
    #        trainimg_path.append(img_path + ',' + image_names)
    #    if i % 1000 == 0 or i == total:
    #        print('Processing image list %i of %i\r' % (i, total))
    #    i += 1
    #random.shuffle(trainimg_path)
 

    batch_size = 128
    

    #img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,
    #224))
    #x = tf.keras.preprocessing.image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = tf.keras.applications.vgg16.preprocess_input(x)

    train_files = tf.data.Dataset.list_files("coco_record/*.tfrecord")
    dataset_train = train_files.interleave(tf.data.TFRecordDataset, cycle_length=4, num_parallel_calls=4)
    dataset_train = dataset_train.shuffle(buffer_size=int(len(trainimg_ids) / batch_size))
    dataset_train = dataset_train.map(_parse_function, num_parallel_calls=cores)
    # padded shape must fit parse output
    dataset_train = dataset_train.padded_batch(batch_size, \
                                                padded_shapes=([None,None], \
                                                [None, None, None], \
                                                [None, None, None]))
    dataset_train = dataset_train.prefetch(batch_size)
    # https://tensorflow.google.cn/api_docs/python/tf/data/Iterator
    iterator = iter(dataset_train)
    count = 0
    # total = int(total / 8)
    bbox_run, images_train, images_decode = [0 for x in range(0,total)],[0 for x in range(0,total)],[0 for x in range(0,total)]
    try:
        while True:
            bbox_run[count], images_train[count], images_decode[count] = iterator.get_next()
            count += 1
    except Exception as exc:
        print("Exception catched as : %s" % exc)
        print('\n遍历结束,迭代次数为',count,'\n')
 
    #验证数据
    imgdex = random.randint(0,count)     #select one image in the batch
    image = images_decode[imgdex]
    image_bbox = bbox_run[imgdex]
    for i in range(image_bbox.shape[0]):
        cv2.rectangle(image, (image_bbox[i][0],image_bbox[i][2]), (image_bbox[i][1],image_bbox[i][3]), (0,255,0), 2)
    plt.imshow(image)

    if not os.path.exists("coco_cache/"):
        os.makedirs("coco_cache/")
        os.makedirs("coco_cache/weights")
    csvlog = tf.keras.callbacks.CSVLogger("coco_cache/traincsv.log", separator=',', append=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint("coco_cache/weights",
                                                 verbose=0, 
                                                 save_best_only=False, 
                                                 save_weights_only=False, 
                                                 save_freq='epoch',
                                                 mode='auto',
                                                 patience=2)

    netmodel = NNAPI("vgg16")
    netmodel.fit(x=images_train, y=bbox_run, 
                 epochs=50,
                 callbacks=[cp_callback,csvlog],
                 validation_data=None)

