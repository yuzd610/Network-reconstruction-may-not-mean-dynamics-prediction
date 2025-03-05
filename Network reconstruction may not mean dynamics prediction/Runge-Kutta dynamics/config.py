import torch

N = 200
g = 1.2
delta_t = 0.01
iterations = 1000

Truncate = 500

set_num = 100

set_num_sect =100

chunk = 125
num_epochs = 40

batch_size = 250




def mean_squared_error(tensor1, tensor2):
    # Check if the input tensor shapes are consistent
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensor shapes are inconsistent")

    # Calculate the square of the subtraction of the corresponding elements at each position
    squared_difference = (tensor1 - tensor2) ** 2

    # sum
    sum_squared_difference = torch.sum(squared_difference)

    # Compute the mean, divided by the number of elements in the tensor
    mean_error = sum_squared_difference / tensor1.numel()

    return mean_error
