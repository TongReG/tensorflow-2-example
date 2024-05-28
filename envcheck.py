# check tensorflow enviroment info
# Run it with "python envcheck.py" in shell
import tensorflow as tf
import platform

# division char...
division = "=====" * 8

# color const... 
COLOR_RED = '\033[91m'  
COLOR_GREEN = '\033[92m'  
COLOR_YELLOW = '\033[93m'  
COLOR_BLUE = '\033[94m'  
COLOR_PURPLE = '\033[95m'
COLOR_CYAN = '\033[96m'
COLOR_WHITE = '\033[97m'
COLOR_RESET = '\033[0m' 
color_set = [COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW,
            COLOR_PURPLE, COLOR_CYAN, COLOR_WHITE]


def colorize(str, color):
    if not color in color_set:
        return str
    
    colorized = {
        COLOR_RED: COLOR_RED + str + COLOR_RESET,
        COLOR_GREEN: COLOR_GREEN + str + COLOR_RESET,
        COLOR_YELLOW: COLOR_YELLOW + str + COLOR_RESET,
        COLOR_BLUE: COLOR_BLUE + str + COLOR_RESET,
        COLOR_PURPLE: COLOR_PURPLE + str + COLOR_RESET,
        COLOR_CYAN: COLOR_CYAN + str + COLOR_RESET,
        COLOR_WHITE: COLOR_WHITE + str + COLOR_RESET,
    }

    return colorized[color]


def printdiv(str):
    print()
    print(colorize(division, COLOR_YELLOW))
    leftRightSpace = int((len(division) - len(str))/2)
    title = " " * leftRightSpace + str + "" * leftRightSpace
    print(colorize(title, COLOR_YELLOW))
    print(colorize(division, COLOR_YELLOW))


def printFeatures(title, val):
    fullstr = title + str(val)
    if val is True:
        print(colorize(fullstr, COLOR_GREEN))
    else:
        print(colorize(fullstr, COLOR_RED))


def printInfo(str):
    print(colorize(str, COLOR_GREEN))


def printError(str):
    print(colorize(str, COLOR_RED))


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
        if name.startswith('is'):
            printFeatures(name + " => ", state)
        else:
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
        print(colorize(paths, COLOR_CYAN))


def showFeatures():
    printdiv("FEATURES")
    tfversion = tf.__version__
    if tfversion < '2.0.0':
        printFeatures("IsMklEnabled ==> ",
                      tf.pywrap_tensorflow.IsMklEnabled())
    else:
        # tf.python has deperacated
        if hasattr(tf, "python"):
            printFeatures("IsBuiltWithNvcc ==> ",
                          tf.python.framework.test_util.IsBuiltWithNvcc())
            printFeatures("IsBuiltWithROCm ==> ",
                          tf.python.framework.test_util.IsBuiltWithROCm())
            printFeatures("IsBuiltWithXLA ==> ",
                          tf.python.framework.test_util.IsBuiltWithXLA())
            printFeatures("IsGoogleCudaEnabled ==> ",
                          tf.python.framework.test_util.IsGoogleCudaEnabled())
            printFeatures("IsMklEnabled ==> ",
                          tf.python.framework.test_util.IsMklEnabled())
        else:
            printFeatures("IsBuiltWithGPUSupport ==> ",
                          tf.test.is_built_with_gpu_support())
            printFeatures("IsBuiltWithCUDA ==> ",
                          tf.test.is_built_with_cuda())
            printFeatures("IsBuiltWithROCm ==> ",
                          tf.test.is_built_with_rocm())
            printFeatures("IsBuiltWithXLA ==> ",
                          tf.test.is_built_with_xla())


def showDevicesFallBack():
    # show CPUs
    Cdv = tf.config.list_physical_devices(device_type='CPU')
    if Cdv != None:
        printdiv("CPU DEVICES")
        for device in Cdv:
            printInfo("{}: {}".format(device.device_type, device.name))
    else:
        printError("Attention: No CPU Devices Available!")
    # show GPUs
    Gdv = tf.config.list_physical_devices(device_type='GPU')
    if Gdv != []:
        printdiv("GPU DEVICES")
        for device in Gdv:
            printInfo("{}: {}".format(device.device_type, device.name))
    else:
        printError("Attention: No GPU Devices Available!")


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

    sysInfo()
    buildInfo()
    versionInfo()
    showFeatures()
    showDevices()
    showPath()
