import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    output_height, output_width = int((input_height + 2*padding - kernel_height) / stride + 1), int((input_width + 2*padding - kernel_width) / stride + 1)
    output_matrix = np.zeros((output_height, output_width))

    if padding > 0:
        padding_h = np.zeros((input_height, padding))
        padding_v = np.zeros((padding, input_width + (2 * padding)))
        padded_horiz = np.concatenate([padding_h, input_matrix, padding_h], axis=1)
        padded_matrix = np.concatenate([padding_v, padded_horiz, padding_v], axis=0)
    else:
        padded_matrix = input_matrix

    for i in range(output_height):
        for j in range(output_width):
            region = padded_matrix[i * stride: i * stride + kernel_height, j * stride: j * stride + kernel_width]
            output_matrix[i][j] = np.sum(region * kernel).item()
    output_matrix = np.array(output_matrix).reshape(output_height, output_width)
    return output_matrix