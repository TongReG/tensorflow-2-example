# check tensorflow enviroment info
# Run it with "python envcheck.py" in shell
import tensorflow as tf
import os
import platform
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用cpu，不注释就是使用gpu

global division


def printdiv(str):
    print()
    print(division)
    leftRightSpace = int((len(division) - len(str))/2)
    print(" " * leftRightSpace + str + "" * leftRightSpace)
    print(division)


def sysInfo():
    printdiv("SYSTEM INFO")
    #
    ostype = platform.system()
    print("OS TYPE: " + ostype)
    print("OS FULLNAME: " + platform.platform())
    print("ARCHITECHURE: " + platform.machine())
    if ostype != 'Linux':
        print("CPU NAME: " + platform.processor())
    if ostype == 'Linux':
        print("LIBC: " + " ".join(platform.libc_ver()))
    print("PYTHON INFO: "
          + platform.python_implementation() + " "
          + platform.python_version()
          + " build by " + platform.python_compiler())


def buildInfo():
    printdiv("TENSORFLOW BUILDINFO")
    tfbuildinfo = tf.sysconfig.get_build_info()
    for name, state in tfbuildinfo.items():
        print(name, "=", state)


def versionInfo():
    printdiv("TENSORFLOW VERSIONIFNO")
    tfversion = tf.__version__
    tfcompilerversion = tf.__compiler_version__
    tfgitversion = tf.__git_version__
    print("TensorFlow Version ==> " + tfversion)
    print("Git Version ==> " + tfgitversion)
    # tf.python has deperacated
    try:
        print("Compiler Version ==> " +
              tf.python.framework.test_util.versions.__compiler_version__)
    except:
        print("Compiler Version ==> " + tfcompilerversion)
    if tf.__cxx_version__ != None:
        tfcxxversion = str(tf.__cxx_version__)
        print("CXX Compiler Version ==> " + tfcxxversion)


def showPath():
    printdiv("tf.__path__")
    for paths in tf.__path__:
        print(paths)


def showFeatures():
    printdiv("FEATURES")
    tfversion = tf.__version__
    if tfversion < '2.0.0':
        print("IsMklEnabled ==> " + str(tf.pywrap_tensorflow.IsMklEnabled()))
    else:
        try:
            print("IsBuiltWithNvcc ==> " +
                  str(tf.python.framework.test_util.IsBuiltWithNvcc()))
            print("IsBuiltWithROCm ==> " +
                  str(tf.python.framework.test_util.IsBuiltWithROCm()))
            print("IsBuiltWithXLA ==> " +
                  str(tf.python.framework.test_util.IsBuiltWithXLA()))
            print("IsGoogleCudaEnabled ==> " +
                  str(tf.python.framework.test_util.IsGoogleCudaEnabled()))
            print("IsMklEnabled ==> " +
                  str(tf.python.framework.test_util.IsMklEnabled()))
        except Exception as e:
            # tf.python has deperacated
            # print("Exception catched as : %s" % e)
            print("IsBuiltWithGPUSupport ==> " +
                  str(tf.test.is_built_with_gpu_support()))
            print("IsBuiltWithCUDA ==> " + str(tf.test.is_built_with_cuda()))
            print("IsBuiltWithROCm ==> " + str(tf.test.is_built_with_rocm()))
            print("IsBuiltWithXLA ==> " + str(tf.test.is_built_with_xla()))


def showDevicesFallBack():
    printdiv("CPU DEVICES")
    Cdv = tf.config.list_physical_devices(device_type='CPU')
    print(Cdv)
    #
    if Cdv != None:
        for device in Cdv:
            print("{}: {}".format(device.device_type, device.name))
    else:
        print("Warning: No CPU Devices Available!")
    printdiv("GPU DEVICES")
    Gdv = tf.config.list_physical_devices(device_type='GPU')
    if Gdv != []:
        for device in Gdv:
            print("GPUDevice ==> ", device)
    else:
        print("No GPU Devices Available!")


def showDevices():
    printdiv("LIST ALL LOCAL DEVICES")
    # 列出所有的本地机器设备
    try:
        local_device_protos = tf.python.client.device_lib.list_local_devices()
        # 打印
        for names in local_device_protos:
            print(names)
    except Exception as e:
        # tf.python has deperacated
        # print("Exception catched as : %s" % e)
        showDevicesFallBack()
        tf.config.list_logical_devices()


# main
if __name__ == '__main__':

    division = "=============================="

    sysInfo()
    buildInfo()
    versionInfo()
    showFeatures()
    showDevices()
    showPath()
