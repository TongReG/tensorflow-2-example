# check tensorflow enviroment
# You'd better run it with "python enviromentcheck.py" in shell but not in
# Visual Studio
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用cpu，不注释就是使用gpu
import tensorflow as tf

def printdiv(str):
    print()
    print("====",str,"====")

printdiv("USING tf.sysconfig.get_build_info()")
tfbuildinfo = tf.sysconfig.get_build_info()
for name,state in tfbuildinfo.items():
    print(name,"==>",state)

printdiv("USING tf.__version__ FLAGS")
tfversion = tf.__version__
tfcompilerversion = tf.__compiler_version__
print("tf.__version__ ==> " + tfversion)
try:
    print("tf.__compiler_version__ ==> " + tf.python.framework.test_util.versions.__compiler_version__)
except Exception as e:
    print("Exception catched as : %s" % e)
    print("tf.__compiler_version__ ==> " + tfcompilerversion)

printdiv("tf.__path__")
for paths in tf.__path__:
    print(paths)

printdiv("USING tf.python/tf.test API")
if tfversion < '2.0.0':
    print("IsMklEnabled ==> " + str(tf.pywrap_tensorflow.IsMklEnabled()))
else:
    try:
        print("IsBuiltWithNvcc ==> " + str(tf.python.framework.test_util.IsBuiltWithNvcc()))
        print("IsBuiltWithROCm ==> " + str(tf.python.framework.test_util.IsBuiltWithROCm()))
        print("IsBuiltWithXLA ==> " + str(tf.python.framework.test_util.IsBuiltWithXLA()))
        print("IsGoogleCudaEnabled ==> " + str(tf.python.framework.test_util.IsGoogleCudaEnabled()))
        print("IsMklEnabled ==> " + str(tf.python.framework.test_util.IsMklEnabled()))
    except Exception as e:
        print("Exception catched as : %s" % e)
        print("IsBuiltWithGPUSupport ==> " + str(tf.test.is_built_with_gpu_support()))
        print("IsBuiltWithCUDA ==> " + str(tf.test.is_built_with_cuda()))
        print("IsBuiltWithROCm ==> " + str(tf.test.is_built_with_rocm()))
        print("IsBuiltWithXLA ==> " + str(tf.test.is_built_with_xla()))

printdiv("USING tf.config API")
Gdv = tf.config.list_physical_devices('GPU')
print("GPUDevices ==> " , Gdv)

printdiv("LIST ALL LOCAL DEVICES")
# 列出所有的本地机器设备
try:
    local_device_protos = tf.python.client.device_lib.list_local_devices()
    # 打印
    for names in local_device_protos:
        print(names)
except Exception as e:
    print("Exception catched as : %s" % e)
    tf.config.list_physical_devices()
    

