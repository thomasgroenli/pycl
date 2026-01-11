import contextlib
import ctypes
from collections import UserList
import numpy as np
import logging
logging.basicConfig()
from .driver import CL

try:
    from math import prod
except ImportError:
    from functools import reduce
    from operator import mul


    def prod(x):
        return reduce(mul, x, 1)

default = CL()


def set_default_driver(driver):
    global default
    default = driver

class Mapper:
    def __init__(self, queue, tensor):
        self.queue = queue
        self.tensor = tensor

    def __getitem__(self, index):
        with self.queue.map(self.tensor, default.CL_MAP_READ) as arr:
            return arr[index]

    def __setitem__(self, index, value):
        with self.queue.map(self.tensor, default.CL_MAP_WRITE) as arr:
            arr[index] = value

class Array(UserList):
    def __init__(self, data):
        self.datatype = type(data[0]._as_parameter_)
        assert all(isinstance(elem._as_parameter_, self.datatype) for elem in data)

        self._buffer = (self.datatype * len(data))(*(elem._as_parameter_ for elem in data))
        self._as_parameter_ = ctypes.cast(self._buffer, ctypes.POINTER(self.datatype))
        super().__init__(data)

class Platform:
    def __init__(self, handle):
        self._as_parameter_ = handle

    @property
    def name(self):
        ret_size = default.size_t()
        default.clGetPlatformInfo(self, default.CL_PLATFORM_NAME, 0, default.NULL, ctypes.byref(ret_size))
        buffer = ctypes.create_string_buffer(ret_size.value)
        default.clGetPlatformInfo(self, default.CL_PLATFORM_NAME, ret_size, buffer, default.NULL)
        return buffer.value.decode('utf-8')

    @staticmethod
    def get_platforms():
        num_platforms = default.cl_uint()
        default.clGetPlatformIDs(num_platforms=ctypes.byref(num_platforms))
        platforms = (default.cl_platform_id * num_platforms.value)()
        default.clGetPlatformIDs(platforms=platforms)
        return Array([Platform(platform) for platform in platforms])

    @staticmethod
    def get_platform(index=0):
        return Platform.get_platforms()[index]

    def create_context(self, devices=None):
        if devices is None:
            devices = self.devices
        return Context.from_devices(devices)

    @property
    def devices(self, device_type=default.CL_DEVICE_TYPE_ALL):
        num_devices = default.cl_uint()
        default.clGetDeviceIDs(self, device_type, num_devices=ctypes.byref(num_devices))
        devices = (default.cl_device_id * num_devices.value)()
        default.clGetDeviceIDs(self, device_type, devices=devices)
        return Array([Device(device) for device in devices])

    def __repr__(self):
        return f"{self.__class__.__name__}[0x{self._as_parameter_.value:0X}]({self.name})"

class Device:
    def __init__(self, handle):
        self._as_parameter_ = handle

    @property
    def name(self):
        ret_size = default.size_t()
        default.clGetDeviceInfo(self, default.CL_DEVICE_NAME, 0, default.NULL, ctypes.byref(ret_size))
        buffer = ctypes.create_string_buffer(ret_size.value)
        default.clGetDeviceInfo(self, default.CL_DEVICE_NAME, ret_size, buffer, default.NULL)
        return buffer.value.decode('utf-8')

    @property
    def platform(self):
        ret_platform = default.cl_platform_id()
        default.clGetDeviceInfo(self, default.CL_DEVICE_PLATFORM, ctypes.sizeof(ret_platform), ctypes.byref(ret_platform), default.NULL)
        return Platform(ret_platform)

    def create_subdevices(self, properties):
        num_devices_ret = default.cl_uint()
        default.clCreateSubDevices(self, properties, 0, default.NULL, ctypes.byref(num_devices_ret))
        devices_out = (default.cl_device_id * num_devices_ret.value)()
        default.clCreateSubDevices(self, properties, num_devices_ret.value, devices_out, default.NULL)
        return Array([Device(device) for device in devices_out])

    def split_cores(self, subdevice_cores):
        properties = (default.cl_device_partition_property * (3 + len(subdevice_cores)))(default.CL_DEVICE_PARTITION_BY_COUNTS,
                                                                                *subdevice_cores,
                                                                                default.CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0)
        return self.create_subdevices(properties)


    def __repr__(self):
        return f"{self.__class__.__name__}[0x{self._as_parameter_.value:0X}]({self.name})"

    def __del__(self):
        default.clReleaseDevice(self)



class LocalMem:
    def __init__(self, bytes):
        self.bytes = bytes
        self._as_parameter_ = default.NULL

class Context:
    def __init__(self, handle):
        self._as_parameter_ = handle

    @property
    def devices(self):
        num_devices = default.cl_uint()
        default.clGetContextInfo(self, default.CL_CONTEXT_NUM_DEVICES, ctypes.sizeof(num_devices), ctypes.byref(num_devices), default.NULL)
        devices = (default.cl_device_id * num_devices.value)()
        default.clGetContextInfo(self, default.CL_CONTEXT_DEVICES, ctypes.sizeof(devices), ctypes.byref(devices),
                                        default.NULL)

        return Array([Device(device) for device in devices])

    def create_queue(self, device, properties=default.NULL):
        return CommandQueue(default.clCreateCommandQueueWithProperties(self, device, properties, default.NULL))

    def create_buffer(self, size, flags=default.CL_MEM_READ_WRITE, host_ptr=default.NULL):
        return Buffer(default.clCreateBuffer(self, flags, size, host_ptr, default.NULL))

    def create_tensor(self, shape, dtype, flags=default.CL_MEM_READ_WRITE, host_ptr=default.NULL):
        size = prod(shape) * np.dtype(dtype).itemsize
        tensor = Tensor(default.clCreateBuffer(self, flags, size, host_ptr, default.NULL))
        tensor.set_shape(shape)
        tensor.set_dtype(dtype)
        return tensor

    def to_tensor(self, array, queue=None):
        tensor = self.create_tensor(array.shape, array.dtype)
        if queue is not None:
            queue.view(tensor)[:] = array
        return tensor

    def create_program(self, sources, build=True):
        if not isinstance(sources, (list, tuple)):
            sources = (sources,)

        encoded = []
        for source in sources:
            if isinstance(source, str):
                source = source.encode('utf-8')
            encoded.append(source)

        n_sources = len(encoded)
        source_buffer = (default.char_p * n_sources)(*(default.char_p(source) for source in encoded))
        length = (default.size_t * n_sources)(*(default.size_t(len(source)) for source in encoded))
        return Program(default.clCreateProgramWithSource(self, n_sources, source_buffer, length, default.NULL))

    @staticmethod
    def from_devices(devices, properties=default.NULL):
        return Context(default.clCreateContext(properties, len(devices), devices, default.NULL, default.NULL, default.NULL))

    def __del__(self):
        if self._as_parameter_ is not None:
            default.clReleaseContext(self)

    def __repr__(self):
        return f"{self.__class__.__name__}[0x{self._as_parameter_.value:0X}]"

class Event:
    def __init__(self, handle):
        self._as_parameter_ = handle

    def wait(self):
        default.clWaitForEvents(1, ctypes.byref(self._as_parameter_))

    def profile(self):
        self.wait()
        start = default.cl_ulong()
        end = default.cl_ulong()
        default.clGetEventProfilingInfo(self, default.CL_PROFILING_COMMAND_START, ctypes.sizeof(default.cl_ulong), ctypes.byref(start), default.NULL)
        default.clGetEventProfilingInfo(self, default.CL_PROFILING_COMMAND_END, ctypes.sizeof(default.cl_ulong), ctypes.byref(end), default.NULL)
        return (end.value - start.value)/1e9

    def __del__(self):
        default.clReleaseEvent(self)

class CommandQueue:
    def __init__(self, handle):
        self._as_parameter_ = handle

    @property
    def context(self):
        ret_context = default.cl_context()
        default.clGetCommandQueueInfo(self, default.CL_QUEUE_CONTEXT, ctypes.sizeof(ret_context), ctypes.byref(ret_context), default.NULL)
        default.clRetainContext(ret_context)
        return Context(ret_context)

    @property
    def device(self):
        ret_device = default.cl_device_id()
        default.clGetCommandQueueInfo(self, default.CL_QUEUE_DEVICE, ctypes.sizeof(ret_device), ctypes.byref(ret_device), default.NULL)
        default.clRetainDevice(ret_device)
        return Device(ret_device)

    def __del__(self):
        if self._as_parameter_ is not None:
            default.clReleaseCommandQueue(self)

    def __repr__(self):
        return f"{self.__class__.__name__}[0x{self._as_parameter_.value:0X}]"

    def finish(self):
        default.clFinish(self)

    def flush(self):
        default.clFlush(self)

    @contextlib.contextmanager
    def map(self, tensor, mode=default.CL_MAP_READ):
        ptr = None
        try:
            ptr = default.clEnqueueMapBuffer(self, tensor, default.CL_TRUE, mode, 0, tensor.size, 0,
                                                    default.NULL, default.NULL, default.NULL)
            arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_byte)), shape=(tensor.size,)).view(tensor.dtype).reshape(tensor.shape)
            if mode == default.CL_MAP_READ:
                arr.flags.writeable = False
            yield arr

        finally:
            if ptr is not None:
                default.clEnqueueUnmapMemObject(self, tensor, ptr, 0, default.NULL, default.NULL)

    def fill(self, buffer, value):
        pattern = np.array([value], dtype=buffer.dtype)
        default.clEnqueueFillBuffer(self, buffer, pattern.ctypes.data, pattern.itemsize, 0, buffer.size, 0, default.NULL, default.NULL)

    def view(self, tensor):
        return Mapper(self, tensor)


    @property
    def run(self):
        class Runner:
            def __getitem__(_, item):
                event = Event(default.cl_event())
                if isinstance(item, tuple):
                    global_item_size, local_item_size = item
                    global_item_size = default.size_t(global_item_size)
                    local_item_size = default.size_t(local_item_size)
                    return lambda kernel: (event, default.clEnqueueNDRangeKernel(self, kernel, 1, default.NULL, ctypes.byref(global_item_size), ctypes.byref(local_item_size), 0, default.NULL, ctypes.byref(event._as_parameter_)))[0]

                else:
                    global_item_size = default.size_t(item)
                    return lambda kernel: (event, default.clEnqueueNDRangeKernel(self, kernel, 1, default.NULL, ctypes.byref(global_item_size), default.NULL, 0, default.NULL, ctypes.byref(event._as_parameter_)))[0]

        return Runner()

class Buffer:
    def __init__(self, handle):
        self._as_parameter_ = handle

    @property
    def context(self):
        ret_context = default.cl_context()
        default.clGetMemObjectInfo(self, default.CL_MEM_CONTEXT, ctypes.sizeof(ret_context), ctypes.byref(ret_context), default.NULL)
        default.clRetainContext(ret_context)
        return Context(ret_context)

    @property
    def size(self):
        ret_size = default.size_t()
        default.clGetMemObjectInfo(self, default.CL_MEM_SIZE, ctypes.sizeof(ret_size), ctypes.byref(ret_size), default.NULL)
        return ret_size.value

    def __del__(self):
        if self._as_parameter_ is not None:
            default.clReleaseMemObject(self)

class Program:
    def __init__(self, handle):
        self._as_parameter_ = handle

    def __del__(self):
        if self._as_parameter_ is not None:
            default.clReleaseProgram(self)

    def __getitem__(self, item):
        return self.create_kernel(item)

    def build(self, devices=default.NULL):
        if devices == default.NULL:
            num_devices = 0
        else:
            num_devices = len(devices)
        try:
            options = ctypes.create_string_buffer(b"-cl-fast-relaxed-math")
            default.clBuildProgram(self, num_devices, devices, options, default.NULL, default.NULL)
        except Exception as exc:
            for device in devices:
                print(f"=== BUILD LOG ({device}) ===")
                print(self.get_log(device))
            raise exc

    def get_log(self, device):
        ret_size = default.size_t()
        default.clGetProgramBuildInfo(self, device, default.CL_PROGRAM_BUILD_LOG, 0, default.NULL, ctypes.byref(ret_size))

        buffer = ctypes.create_string_buffer(ret_size.value)
        default.clGetProgramBuildInfo(self, device, default.CL_PROGRAM_BUILD_LOG, ret_size, buffer, default.NULL)
        return buffer.value.decode('utf-8')

    def create_kernel(self, kernel_name):
        if isinstance(kernel_name, str):
            kernel_name = kernel_name.encode('utf-8')
        return Kernel(default.clCreateKernel(self, kernel_name, default.NULL))

    @property
    def context(self):
        ret_context = default.cl_context()
        default.clGetProgramInfo(self, default.CL_PROGRAM_CONTEXT, ctypes.sizeof(ret_context), ctypes.byref(ret_context), default.NULL)
        default.clRetainContext(ret_context)
        return Context(ret_context)

    @property
    def devices(self):
        num_devices = default.cl_uint()
        default.clGetProgramInfo(self, default.CL_PROGRAM_NUM_DEVICES, ctypes.sizeof(num_devices), ctypes.byref(num_devices),
                                        default.NULL)
        devices = (default.cl_device_id * num_devices.value)()
        default.clGetProgramInfo(self, default.CL_PROGRAM_DEVICES, ctypes.sizeof(devices), ctypes.byref(devices),
                                        default.NULL)

        return Array([Device(device) for device in devices])


    def __repr__(self):
        return f"{self.__class__.__name__}[0x{self._as_parameter_.value:0X}]"

class Kernel:
    def __init__(self, handle):
        self._as_parameter_ = handle

    def __del__(self):
        if self._as_parameter_ is not None:
            default.clReleaseKernel(self)

    def __getitem__(self, item):
        return

    def __repr__(self):
        return f"{self.__class__.__name__}[0x{self._as_parameter_.value:0X}]"

    def set_arg(self, index, value):
        if isinstance(value, (Buffer, Tensor)):
            size = ctypes.sizeof(default.cl_mem)
            value = ctypes.byref(value._as_parameter_)
        elif isinstance(value, LocalMem):
            size = value.bytes
            value = default.NULL
        else:
            size = ctypes.sizeof(value)
            value = ctypes.byref(value)
        default.clSetKernelArg(self, index, size, value)

    def set_args(self, *args):
        for i, arg in enumerate(args):
            self.set_arg(i, arg)

class Tensor(Buffer):
    def __init__(self, handle):
        super().__init__(handle)
        self.shape = None
        self.dtype = None

    def set_shape(self, shape):
        self.shape = shape

    def set_dtype(self, dtype):
        self.dtype = dtype

    def __str__(self):
        return f"{self.__class__.__name__}[0x{self._as_parameter_.value:0X}](shape={self.shape}, dtype={np.dtype(self.dtype)})"
