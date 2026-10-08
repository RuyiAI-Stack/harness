"""AST-only discovery fixture; not an executable correctness benchmark."""
import triton
import triton.language as tl


@triton.jit
def relu_and_mul_kernel(x, y, out):
    a = tl.load(x)
    b = tl.load(y)
    tl.store(out, tl.maximum(a, 0) * b)


def relu_and_mul(x, y):
    return relu_and_mul_kernel(x, y, x)
