# Extra: Train with celeaar on pytorch using SegNET
# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data

# Torchvision
import torchvision
import torchvision.transforms as transforms
from torchvision import utils
from torch.utils.data import Dataset

# Matplotib
import matplotlib.pyplot as plt
# OS
import os
import argparse
# opencv
import cv2
# PIL
from PIL import Image
# sklearn
from skimage import io, transform

# Set random seed for reproducibility
SEED = 123

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)#为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
def __getitem__(self, index):   #自定义dataset要覆写
        raise NotImplementedError
def __len__(self):
        raise NotImplementedError

#def print_model(encoder, decoder): 打印网络结构
#    print("============== Encoder ==============")
#    print(encoder)
#    print("============== Decoder ==============")
#    print(decoder)
#    print("")
def create_model():
    autoencoder = Autoencoder()
    # print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder

def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)  #它封装了Tensor，并整合了反向传播的相关实现(tensor变成variable之后才能进行反向传播求梯度
                        #用变量.backward()进行反向传播之后,var.grad中保存了var的梯度)
def imshow(img):
    npimg = img.cpu().numpy() #numpy()将tensor转换为numpy：注意cuda上面的变量类型只能是tensor，不能是其他
    plt.axis('off')#关坐标轴
    plt.imshow(np.transpose(npimg, (1, 2, 0)))# 转置相对于（0，1，2）
    plt.show()
    #plt.imshow()函数负责对图像进行处理，并显示其格式，而plt.show()则是将plt.imshow()处理后的函数显示出来。

#class img2Gray(object):
#    def __call__(self, tensor):
#        # TODO: make efficient
#        R = tensor[0]
#        G = tensor[1]
#        B = tensor[2]
#        tensor[0]=0.299*R+0.587*G+0.114*B
#        tensor = tensor[0]
#        tensor = tensor.view(1,160,160)
#        return tensor

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32] 3*178*218 3*160*160
        # Output size: [batch, 3, 32, 32] 3*160*160
        batchNorm_momentum = 0.1
        # 一个用于运行过程中均值和方差的一个估计参数（我的理解是一个稳定系数，类似于SGD中的momentum的系数）

# ========================//encoder//==========================================
        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # same
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        #在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理
        #这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        #self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        #self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        #self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

# ===========================//decoder//====================================
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        #self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        #self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        #self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.bn11d = nn.BatchNorm2d(3, momentum=batchNorm_momentum)

    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        #x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x11, kernel_size=2, stride=2, return_indices=True)  # 80*80
        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)  # 40*40
        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        #x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x32, kernel_size=2, stride=2, return_indices=True)  # 20*20
        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)  # 10*10
        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)  # 5*5

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)  # 10*10
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))
        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)  # 20*20
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))
        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)  # 40*40
        #x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x3d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))
        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)  # 80*80
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))
        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)  # 160*160
        #x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = F.relu(self.bn11d(self.conv11d(x1d)))

        return x11d

class faceData(Dataset):  # 自定义继承自Dataset
    def __init__(self, root_dir, mask_dir, transform):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # face目录
        self.mask_dir = mask_dir
        self.transform = transform # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件
        self.images2 = os.listdir(self.mask_dir)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        if 18000 > index >= 9000: # fake
            image_face_name = self.images[index]
            image_mask_name = self.images2[index - 9000]
            image_face_path = os.path.join(self.root_dir, image_face_name)
            image_mask_path = os.path.join(self.mask_dir, image_mask_name)
            img_face = cv2.imread(image_face_path)
            img_mask = cv2.imread(image_mask_path)

            label_face = "fake_face"
            label_mask = "fake_mask"
            # img_face = img_face.transpose(2, 0, 1)
            # img_mask = img_mask.transpose(2, 0, 1)
            if self.transform:
                img_mask = self.transform(img_mask)  # 对样本进行变换
                img_face = self.transform(img_face)
            return img_face, img_mask

        elif 9000 > index >= 0: # true
            image_face_name = self.images[index]
            image_face_path = os.path.join(self.root_dir, image_face_name)
            image_mask_path = os.path.join(self.mask_dir, "true_mask.jpg") # 真人脸痕迹图都是全黑, 只做了一张
            img_face = cv2.imread(image_face_path)
            img_mask = cv2.imread(image_mask_path)

            label_face = "true_face"
            label_mask = "true_mask"
            if self.transform:
                img_face = self.transform(img_face)# 对样本进行变换
                img_mask = self.transform(img_mask)
            return img_face, img_mask

        else:
            print(index)
            print("index wrong!")
            return -1

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    #argparse 模块定义解析命令行参数，命令行参数其实也是应用在程序中的参数，只是为了更方便他人使用程序而设置。
    parser.add_argument("--valid", action="store_true", default=False, help="Perform validation only.")#仅执行验证
    args = parser.parse_args()

    # Create model
    autoencoder = create_model()
    # Load data
    trans = transforms.Compose(#组合
        [transforms.ToPILImage(), transforms.CenterCrop(160), transforms.ToTensor()])#[h,w]

    trainset = faceData("F:\\python_study\\autoencoder\\sx\\data\\train\\", "F:\\python_study\\autoencoder\\sx\\data\\train_mask\\", trans)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    #该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch
    #size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入，

#    testset =
#    torchvision.datasets.ImageFolder('F:\\python_study\\autoencoder\\sx\\data\\test',
#    transform=transform)
#    testloader = torch.utils.data.DataLoader(testset,
#    batch_size=10,shuffle=False, num_workers=2)

    if args.valid:
        print("Abolish")
    #    print("Loading checkpoint...")
    #    autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
    #    dataiter = iter(testloader)
    #    images, labels = dataiter.next()
    #    #print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in
    #    range(20)))
    #    imshow(torchvision.utils.make_grid(images))
    #    images = Variable(images.cuda())
    #    decoded_imgs = autoencoder(images)[1]
    #    imshow(torchvision.utils.make_grid(decoded_imgs.data))
        exit(0)

    # Define an optimizer and criterion
    criterion = nn.BCELoss()
    #BCELoss是二分类的交叉熵损失
    optimizer = optim.Adam(autoencoder.parameters())
    # 优化器就是需要根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用，这也是机器学习里面最一般的方法论。
    # 从优化器的作用出发，要使得优化器能够起作用，需要主要两个东西：
    # 1.优化器需要知道当前的网络或者别的什么模型的参数空间，这也就是为什么在训练文件中，正式开始训练之前需要将网络的参数放到优化器里面
    # 例如 optimizer_G = Adam(model_G.parameters(), lr=train_c.lr_G)
    # 2.需要知道反向传播的梯度信息

    #LOSS LINE
    Loss_list = []
    x2 = range(0, 30)

    for epoch in range(30):
        running_loss = 0.0

        for i, inputs in enumerate(trainloader, 0):
            face, mask = inputs
            face = get_torch_vars(face) # variable
            mask = get_torch_vars(mask)

            # ============ Forward ============
            # with torch.no_grad():
            outputs = autoencoder(face)
            outputs[outputs < 0.0] = 0.0
            outputs[outputs > 1.0] = 1.0
            loss = criterion(outputs, mask)

            # ============ Backward ============
            # if 20 > epoch > 9:
            #    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0003)
            # elif 30 > epoch > 19:
            #    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)
            optimizer.zero_grad()
            #optimzier使用之前需要zero清零一下，因为如果不清零，那么使用的这个grad就得同上一个mini-batch有关
            loss.backward()
            #误差反向传播计算参数梯度
            optimizer.step()

            # ============ Logging ============
            running_loss += loss.data
            if i % 4500 == 4499:
                print(autoencoder.parameters())
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 4500))
                Loss_list.append(running_loss / 4500)
                running_loss = 0.0

                img_1 = outputs.cpu().detach().numpy() * 255
                img_1 = img_1.astype('uint8')
                for tmp in range(0, 4):
                     img_2 = img_1[tmp]
                     # print(type(img_2))
                     # print(img_2.shape)
                     plt.imshow(img_2[0])
                     plt.savefig("F:\\python_study\\autoencoder\\PyTorch-CIFAR-10-autoencoder-master\\PyTorch-CIFAR-10-autoencoder-master"
                                 "\\peek\\" + str(epoch) + "_" + str(tmp) + ".jpg")
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    # plt.show()
    plt.plot(x2, Loss_list)
    plt.savefig("F:\\python_study\\autoencoder\\PyTorch-CIFAR-10-autoencoder-master"
                "\\PyTorch-CIFAR-10-autoencoder-master\\" + "line" + ".jpg")
    print('Finished Training')
    if not os.path.exists('./weights'):  # make dir
        os.mkdir('./weights')
    torch.save(autoencoder, "./weights/autoencoder_all.pkl")
    print('Save Model successful')
if __name__ == '__main__':
    main()
