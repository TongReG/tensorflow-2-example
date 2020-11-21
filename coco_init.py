#基础4 多种Net On MSCOCO 2017 Dataset Example
#你需要先下载COCO 2017数据集并放入本仓库根目录下，或手动更改path
#训练之前请务必运行本文件以生成TFRECORD文件集，大小约15GB
#训练过程需读取TFRECORD，以加载数据
#http://images.cocodataset.org/zips/train2017.zip
#http://images.cocodataset.org/annotations/annotations_trainval2017.zip

#http://images.cocodataset.org/zips/val2017.zip
#http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

#http://images.cocodataset.org/zips/test2017.zip
#http://images.cocodataset.org/annotations/image_info_test2017.zip
import os
import time
import math
import random
from multiprocessing import Process, Queue, cpu_count, freeze_support
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉可以调用GPU，不注释时使用CPU
import pycocotools.coco

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
val_keypoints = 'annotations/person_keypoints_val2017.json'
val_instances = 'annotations/instances_val2017.json'
val_captions = 'annotations/captions_val2017.json'
#COCO训练集的图像数据保存的目录
img_path = 'train2017/'
val_path = 'val2017/'
catas_dict = {}
imgrcg_box = {}
cores = int(cpu_count() / 2)   #定义用到的CPU处理的核心数
if cores >= 6:
    cores = 6
max_num = 2048   #每个TFRECORD文件包含的最多的图像数

#COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#                'giraffe',
#                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
#                'frisbee',
#                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
#                'bowl',
#                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',*
#                'potted plant', 'bed', 'dining table', 'toilet', 'tv',
#                'laptop',
#                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#                'oven',
#                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

#COCO_LABEL_MAP = { 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
#                   9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17:
#                   16,
#                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25:
#                  24,
#                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36:
#                  32,
#                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44:
#                  40,
#                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53:
#                  48,
#                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61:
#                  56,
#                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73:
#                  64,
#                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81:
#                  72,
#                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90:
#                  80}

#annotations字段是包含多个annotation实例的一个数组.
#segmentation格式取决于这个实例
#是一个单个的对象（即iscrowd=0，使用polygons格式）
#还是一组对象（即iscrowd=1，使用RLE格式）
#另外，每个对象都会有一个矩形框"bbox":[x,y,width,height]
#矩形框左上角坐标和框的长宽会以数组的形式.

#一段解析polygon格式的segmentation并且显示多边形的示例代码
def readplygn():
    if ['iscrowd'] == 0:
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

#把图像以及对应的检测框，类别等数据保存到TFRECORD
def make_example(image, height, width, label, bbox, filename):
    colorspace = b'RGB'
    channels = 3
    img_format = b'JPEG'
    return tf.train.Example(features=tf.train.Features(feature={'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                     'height' : tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                     'width' : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                     'channels' : tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
                     'colorspace' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[colorspace])),
                     'img_format' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_format])),
                     'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                     'bbox_xmin' : tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[0])),
                     'bbox_ymin' : tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[1])),
                     'bbox_xmax' : tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[2])),
                     'bbox_ymax' : tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[3])),
                     'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename]))}))

#定义多进程函数用于生成TFRECORD文件
def gen_tfrecord(trainrecords, targetfolder, num, queue, imgrcg_box, catas_dict, all_cata_dict):
    records_file_num = num
    file_num = 0
    total_num = len(trainrecords)
    pid = os.getpid()
    queue.put((pid, file_num))
    writer = tf.io.TFRecordWriter(targetfolder + "train_" + str(records_file_num) + ".tfrecord")
    for record in trainrecords:
        file_num += 1
        fields = record.split(',')
        img = cv2.imread(fields[0] + fields[1])
        height, width, _ = img.shape
        img_jpg = cv2.imencode('.jpg', img)[1].tobytes()
        fname = fields[1]
        #如果用dict[key]这个读取会报KeyError异常，
        #dict.get方法主要是提供一个取不到对应key的value就返回默认值的功能
        bbox = imgrcg_box.get(fname)
        if bbox:
            bbox[1] = [item for item in bbox[1]]   #xmin
            bbox[3] = [item for item in bbox[3]]   #xmax
            bbox[2] = [item for item in bbox[2]]  #ymin
            bbox[4] = [item for item in bbox[4]]  #ymax
            catnames = [catas_dict[item] for item in bbox[0]]
            label = [all_cata_dict[item] for item in catnames]
            extra = make_example(img_jpg, height, width, label, bbox[1:], fname.encode())
            writer.write(record=extra.SerializeToString())
        else: continue
        #每写入100条记录，向父进程发送消息，报告进度
        if file_num % 100 == 0:
            queue.put((pid, file_num))
        if file_num % max_num == 0:
            writer.close()
            records_file_num += 1
            writer = tf.io.TFRecordWriter(targetfolder + "train_" + str(records_file_num) + ".tfrecord")
    writer.close()        
    queue.put((pid, file_num))

#定义多进程处理
def process_in_queues(namelist, cores, targetfolder):
    total_files_num = len(namelist)
    perprocess_files_num = int(total_files_num / cores)
    files_for_process_list = []
    for i in range(cores - 1):
        files_for_process_list.append(namelist[i * perprocess_files_num : (i + 1) * perprocess_files_num])
    files_for_process_list.append(namelist[(cores - 1) * perprocess_files_num:])
    files_number_list = [len(l) for l in files_for_process_list]
    
    perprocess_tffiles_num = math.ceil(perprocess_files_num / max_num)
    
    queues_list = []
    processes_list = []
    for i in range(cores):
        queues_list.append(Queue())
        #queue = Queue()
        processes_list.append(Process(target=gen_tfrecord, 
                                      args=(files_for_process_list[i],targetfolder,
                                      perprocess_tffiles_num * i + 1,queues_list[i],
                                      imgrcg_box,catas_dict,all_cata_dict)))
 
    for p in processes_list:
        Process.start(p)
 
    #父进程循环查询队列的消息，并且每0.5秒更新一次
    while(True):
        try:
            total = 0
            progress_str = ''
            for i in range(cores):
                msg = queues_list[i].get()
                total += msg[1]
                progress_str += 'PID' + str(msg[0]) + ':' + str(msg[1]) + '/' + str(files_number_list[i]) + '|'
            progress_str += '\r'
            print(progress_str)
            if total == total_files_num:
                for p in processes_list:
                    p.terminate()
                    p.join()
                break
            time.sleep(0.5)
        except Exception as e:
            print("\nException catched as : %s" % e)
            print("Queue Process Failed. End execution.")
            break
    return total

if __name__ == '__main__':
    freeze_support()

    # Default load is instances, you can change the path to keypoints or
    # captions.
    coco_train = pycocotools.coco.COCO(path_instances)
    coco_val = pycocotools.coco.COCO(val_instances)

    COCO_CLASSES = coco_train.dataset['categories']
    
    train_ids = list(coco_train.imgToAnns.keys())
    val_ids = list(coco_val.imgToAnns.keys())
    #if len(self.ids) == 0: # 如果没有标签或者不需要GT，则直接使用image
    #    train_ids = list(coco_train.imgs.keys())

    # Display COCO categories and supercategories.  id ==> name
    catas = coco_train.loadCats(coco_train.getCatIds())
    for cata in catas:
        catas_dict[cata['id']] = cata['name']
  
    # 获取COCO数据集中所有图像的ID,构建训练集文件列表，里面的每个元素是路径名+图片文件名
    trainimg_ids = coco_train.getImgIds()
    valimg_ids = coco_val.getImgIds()

    print('trainimg_IDNUMBERS =>',len(trainimg_ids),'valimg_IDNUMBERS =>',len(valimg_ids))

    train_images_filenames = os.listdir(img_path)
    val_images_filenames = os.listdir(val_path)

    # 查找训练集的图片是否都有对应的ID，并保存到一个列表中
    trainimg_path, valimg_path = [], []
    i = 1
    num_train = len(train_images_filenames)
    num_val = len(val_images_filenames)
    print('trainimg_NUMBERS =>',num_train,'valimg_NUMBERS =>',num_val)
    for image_names in train_images_filenames:
        if int(image_names[:-4]) in trainimg_ids:
            trainimg_path.append(img_path + ',' + image_names)
        if i % 100 == 0 or i == num_train:
            print('Processing image list %i of %i\r' % (i, num_train))
        i += 1
    random.shuffle(trainimg_path)
 
    all_catas = set()   #保存目标检测所有的类别，COCO共定义了90个类别，只有80个类别有检测数据
    #获取每个图像的目标检测框的数据并保存
    for path in trainimg_path:
        boxes = [[],[],[],[],[]]
        fname = path.split(',')[1]
        imgid = int(fname[:-4])
        annIds_nocrowd = coco_train.getAnnIds(imgIds=imgid, iscrowd=False)
        annoations = coco_train.loadAnns(annIds_nocrowd)
        for ann in annoations:
            bbox = ann['bbox']
            x_min = int(bbox[0])
            x_max = int(bbox[0] + bbox[2])
            y_min = int(bbox[1])
            y_max = int(bbox[1] + bbox[3])
            all_catas.add(catas_dict[ann['category_id']])
            boxes[0].append(ann['category_id'])
            boxes[1].append(x_min)
            boxes[2].append(y_min)
            boxes[3].append(x_max)
            boxes[4].append(y_max)
        imgrcg_box[fname] = boxes
        if fname in imgrcg_box:
            print('imgrcg_box',fname,'=>',imgrcg_box[fname])
        else: print('imgrcg_box',fname,'Does not exist!!')
 
    #获取有目标检测数据的80个类别的名称 name => id
    all_cata_list = list(all_catas)
    all_cata_dict = {}
    for i in range(len(all_cata_list)):
        all_cata_dict[all_cata_list[i]] = i
    print(all_cata_dict)

    print('\nTotal CPU Threads =>',cpu_count(),'\n')

    if not os.path.exists('coco_record/'):
        os.makedirs('coco_record/')

    print('Start processing train data using %i CPU Threads:' % cores,'\n')
    startime = time.time()        
    total_processed = process_in_queues(trainimg_path, cores, targetfolder='coco_record/')
    endtime = time.time()
    print('\nProcess finish, total process %i images in %i seconds.' % (total_processed, int(endtime - startime)), end='')


