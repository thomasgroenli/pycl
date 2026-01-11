import cl
import numpy as np

driver = cl.helpers.default
source = """
    __kernel void vector_add(__global const float* A, __global const float* B, __global float* C, const unsigned int N) {
    int gid = get_global_id(0);
    int stride = get_global_size(0);

    for (int i = gid; i < N; i += stride)  {
        C[i] = A[i] + B[i];
    }
}
"""

# Find first available platform
platform = cl.helpers.Platform.get_platform(0)

# Create context using all devices
context = platform.create_context(platform.devices)

# Create queue for first device in context
queue = context.create_queue(context.devices[0])

# Create and build program
program = context.create_program(source)
program.build()

# Create kernel from program
kernel = program.create_kernel('vector_add')

# Create input numpy arrays and transfer to device through queue
N = 10000000

a = context.to_tensor(np.ones(N,dtype=np.float32), queue)
b = context.to_tensor(np.ones(N,dtype=np.float32), queue)

# Create empty output tensor
c = context.create_tensor(a.shape, dtype=a.dtype)

# Run kernel
kernel.set_args(a, b, c, driver.cl_int(N))
queue.run[1024, 1024](kernel)

# View result through queue (optionally [:].copy() to transfer to host)
print(queue.view(c)[:])

