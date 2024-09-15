import torch

N = 200
g = 2
delta_t = 0.01
iterations = 1000

Truncate = 500

set_num = 100

set_num_sect = 100

chunk = 125

num_epochs = 50

batch_size = 250




def mean_squared_error(tensor1, tensor2):
    # 检查输入张量的形状是否一致
    if tensor1.shape != tensor2.shape:
        raise ValueError("输入张量形状不一致")

    # 计算每个位置对应元素相减后的平方
    squared_difference = (tensor1 - tensor2) ** 2

    # 求和
    sum_squared_difference = torch.sum(squared_difference)

    # 计算平均值，除以张量的元素数量
    mean_error = sum_squared_difference / tensor1.numel()

    return mean_error
