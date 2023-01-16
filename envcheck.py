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
    print("TensorFlow Version ==> " + tfversion)
    if hasattr(tf, "__git_version__"):
        tfgitversion = tf.__git_version__
        print("Git Version ==> " + tfgitversion)
    tfcompilerversion = tf.__compiler_version__
    # tf.python has deperacated
    if hasattr(tf, "python"):
        print("Compiler Version ==> " +
              tf.python.framework.test_util.versions.__compiler_version__)
    else:
        print("Compiler Version ==> " + tfcompilerversion)
    if hasattr(tf, '__cxx_version__'):
        tfcxxversion = str(tf.__cxx_version__)
        print("CXX Compiler Version ==> " + tfcxxversion)
    if hasattr(tf, "__cxx11_abi_flag__"):
        tfcxxflag = str(tf.__cxx11_abi_flag__)
        print("CXX Flag ==> " + tfcxxflag)


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
        # tf.python has deperacated
        if hasattr(tf, "python"):
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
        else:
            print("IsBuiltWithGPUSupport ==> " +
                  str(tf.test.is_built_with_gpu_support()))
            print("IsBuiltWithCUDA ==> " + str(tf.test.is_built_with_cuda()))
            print("IsBuiltWithROCm ==> " + str(tf.test.is_built_with_rocm()))
            print("IsBuiltWithXLA ==> " + str(tf.test.is_built_with_xla()))


def showDevicesFallBack():
    # show CPUs
    printdiv("CPU DEVICES")
    Cdv = tf.config.list_physical_devices(device_type='CPU')
    if Cdv != None:
        for device in Cdv:
            print("{}: {}".format(device.device_type, device.name))
    else:
        print("Warning: No CPU Devices Available!")
    # show GPUs
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
    # tf.python has deperacated
    if hasattr(tf, "python"):
        local_device_protos = tf.python.client.device_lib.list_local_devices()
        # 打印
        for names in local_device_protos:
            print(names)
    else:
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
