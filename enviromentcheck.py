# check tensorflow enviroment
# You'd better run it with "python enviromentcheck.py" but not in Visual Studio 
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用cpu，不注释就是使用gpu
import tensorflow as tf
 
tfversion = tf.__version__
print("tf.__version__ ==> " + tfversion)

print("tf.__compiler_version__ ==> " + tf.python.framework.test_util.versions.__compiler_version__)

for paths in tf.__path__:
    print(paths)

if tfversion < '2.0':
    print(tf.pywrap_tensorflow.IsMklEnabled())
else:
    print("IsBuiltWithNvcc ==> " + str(tf.python.framework.test_util.IsBuiltWithNvcc()))
    print("IsBuiltWithROCm ==> " + str(tf.python.framework.test_util.IsBuiltWithROCm()))
    print("IsBuiltWithXLA ==> " + str(tf.python.framework.test_util.IsBuiltWithXLA()))
    print("IsGoogleCudaEnabled ==> " + str(tf.python.framework.test_util.IsGoogleCudaEnabled()))
    print("IsMklEnabled ==> " + str(tf.python.framework.test_util.IsMklEnabled()))

tf.config.list_physical_devices('GPU')
# 列出所有的本地机器设备
local_device_protos = tf.python.client.device_lib.list_local_devices()
# 打印
for names in local_device_protos:
    print(names)
