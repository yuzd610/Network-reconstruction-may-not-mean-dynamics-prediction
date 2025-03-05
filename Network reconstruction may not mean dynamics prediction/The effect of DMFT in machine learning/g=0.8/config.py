import torch

N = 1000
g = 0.8
delta_t = 0.01
iterations = 1000

Truncate = 300

set_num = 800

set_num_sect = 800

chunk = 100

num_epochs = 25

batch_size = 250




def mean_squared_error(tensor1, tensor2):
    # Check if the input tensor shapes are consistent
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensor shapes are inconsistent")

    # Calculate the square of the subtraction of the corresponding elements at each position
    squared_difference = (tensor1 - tensor2) ** 2

    # Sum
    sum_squared_difference = torch.sum(squared_difference)

    # Calculate the average, dividing by the number of elements in the tensor
    mean_error = sum_squared_difference / tensor1.numel()

    return mean_error
