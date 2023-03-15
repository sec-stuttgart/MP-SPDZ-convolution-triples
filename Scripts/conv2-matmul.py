import fire
import functools
import itertools
import onnx
import math

def prod(iterable):
    return functools.reduce(lambda x, y: x * y, iterable, 1)

def matmul_dimension_for(input_depth, output_height, output_width, output_depth, kernel_height, kernel_width):
    return output_height * output_width, kernel_height * kernel_width * input_depth, output_depth

def main(model, matmul_shape=(128,128,128), verbose=False, summary=False, exclude1x1=False):
    model = onnx.load(model)
    shape_model = onnx.shape_inference.infer_shapes(model)

    shapes = {}
    for n in itertools.chain(shape_model.graph.value_info, shape_model.graph.input):
        shapes[n.name] = tuple(dim.dim_value for dim in n.type.tensor_type.shape.dim)

    matmuls = 0
    convs = {}

    for n in model.graph.node:
        if n.op_type == "Conv":
            input_shape = shapes[n.input[0]]
            output_shape = shapes[n.name]

            for a in n.attribute:
                if a.name == "kernel_shape":
                    kernel_shape = tuple(a.ints)
                elif a.name == "strides":
                    strides = tuple(a.ints)

            if exclude1x1 and kernel_shape == (1, 1):
                continue

            _, input_depth, input_height, input_width = input_shape
            _, output_depth, output_height, output_width = output_shape 
            kernel_height, kernel_width = kernel_shape

            matmul_dimensions = matmul_dimension_for(input_depth, output_height, output_width, output_depth, kernel_height, kernel_width)

            matmuls_for_conv = prod(math.ceil(dim / shape) for dim, shape in zip(matmul_dimensions, matmul_shape))

            if verbose:
                print(f"Conv \"{n.name}\" ({input_height}x{input_width}x{input_depth} -> {output_height}x{output_width}x{output_depth} with {kernel_height}x{kernel_width} kernel and {strides[0]}x{strides[1]} stride; equivalent matmul is {matmul_dimensions[0]}x{matmul_dimensions[1]} * {matmul_dimensions[1]}x{matmul_dimensions[2]}) requires {matmuls_for_conv} matmuls")

            if summary:
                try:
                    old_matmuls_for_conv, count = convs[input_shape, output_shape, kernel_shape, strides]
                    assert old_matmuls_for_conv == matmuls_for_conv
                except KeyError:
                    count = 0

                convs[input_shape, output_shape, kernel_shape, strides] = (matmuls_for_conv, count + 1)
                    

            matmuls += matmuls_for_conv

    if summary:
        for ((_, input_depth, input_height, input_width), (_, output_depth, output_height, output_width), (kernel_height, kernel_width), strides), (matmuls_for_conv, count) in convs.items():
            plural = " " if count == 1 else "s"
            k, l, d = matmul_dimension_for(input_depth, output_height, output_width, output_depth, kernel_height, kernel_width)
            assert d == output_depth
            print(f"Overall {count} convolution{plural} from {input_height}x{input_width}x{input_depth} to {output_height}x{output_width}x{output_depth} with kernel shape {kernel_height}x{kernel_width} and stride {strides[0]}x{strides[1]} => {count} {k}x{l} * {l}x{d} matmuls => {count} * {matmuls_for_conv} = {count * matmuls_for_conv} matmuls")

        print(f"Total number of {matmul_shape[0]}x{matmul_shape[1]} * {matmul_shape[1]}x{matmul_shape[2]} matmuls: {matmuls}")

    return matmuls

if __name__ == "__main__":
    fire.Fire(main)