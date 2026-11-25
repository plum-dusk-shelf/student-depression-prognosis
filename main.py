import time
import torch

a = torch.tensor([1.2, 4.5, 3.0])
b = torch.tensor([4.0, 0.3, 3.1])

start = time.time()
c = a * b
end = time.time()

print(f'Result tensor: {c};\nTime: {end-start}')

a = a.to('cuda')
b = b.to('cuda')
start = time.time()
c = a * b
end = time.time()

print(f'Time with CUDA: {end-start}')

