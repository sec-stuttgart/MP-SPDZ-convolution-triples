import fire
import functools
import math

def prod(iterable):
    return functools.reduce(lambda x, y: x * y, iterable, 1)

def matmul_dimension_for(input_depth, output_height, output_width, output_depth, kernel_height, kernel_width):
    return output_height * output_width, kernel_height * kernel_width * input_depth, output_depth

def main(output_height, output_width, depth, kernel=3, matmul_shape=(128,128,128), verbose=False):
    try:
        kernel_height, kernel_width = kernel
    except:
        kernel_height, kernel_width = kernel, kernel

    matmul_dimensions = matmul_dimension_for(depth, output_height, output_width, depth, kernel_height, kernel_width)
    matmuls_for_conv = prod(math.ceil(dim / shape) for dim, shape in zip(matmul_dimensions, matmul_shape))

    individual_matmul_dimensions = matmul_dimension_for(1, output_height, output_width, 1, kernel_height, kernel_width)
    individual_matmuls_for_conv = depth * prod(math.ceil(dim / shape) for dim, shape in zip(individual_matmul_dimensions, matmul_shape))

    if matmuls_for_conv < individual_matmuls_for_conv:
        matmuls = matmuls_for_conv

        if verbose:
            k, l, d = matmul_dimensions
            print(f"Depthwise convolution from {output_height}x{output_width}x{depth} with kernel shape {kernel_height}x{kernel_width} => one {k}x{l} * {l}x{d} matmul => {matmuls} matmuls")
    else:
        matmuls = individual_matmuls_for_conv

        if verbose:
            k, l, d = individual_matmul_dimensions
            print(f"Depthwise convolution from {output_height}x{output_width}x{depth} with kernel shape {kernel_height}x{kernel_width} => {depth} {k}x{l} * {l}x{d} matmuls => {matmuls} matmuls")

    return matmuls

if __name__ == "__main__":
    fire.Fire(main)