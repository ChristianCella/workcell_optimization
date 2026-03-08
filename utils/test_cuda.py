''' 
This script is used to check the version of PyTorch and Torchvision installed on the system.
'''

import torch
import torchvision
import fonts

print(f'{fonts.green}PyTorch version: {torch.__version__}{fonts.reset}')
print(f'{fonts.red}Torchvision version: {torchvision.__version__}{fonts.reset}')
print(f'{fonts.yellow}CUDA availability: {torch.cuda.is_available()}{fonts.reset}')
print(f'{fonts.purple}The device for the training is: {torch.cuda.get_device_name(0)}{fonts.reset}')