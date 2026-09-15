import torch

N = 200
g = 2
delta_t = 0.01
iterations = 1000

Truncate = 500

set_num = 100



chunk = 125
num_epochs = 300

batch_size = 250




def mean_squared_error(tensor1, tensor2):
 


    squared_difference = (tensor1 - tensor2) ** 2


    sum_squared_difference = torch.sum(squared_difference)


    mean_error = sum_squared_difference / tensor1.numel()

    return mean_error
