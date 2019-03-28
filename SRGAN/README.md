

```python
import torch
import numpy as np
from torch import nn
```

<img src="https://github.com/haoranD/Paper-Reproduction/blob/master/SRGAN/srgan.png">

<img src="https://github.com/haoranD/Paper-Reproduction/blob/master/SRGAN/conv2d.png">

- https://blog.csdn.net/a132582/article/details/78658155

# First of all, we need to define the `Residual Block`


```python
class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, k=3, p=1):
        super(ResidualBlock, self).__init__()
        
        self.RBnet = nn.Sequential(
            
            nn.Conv2d(input_channels, output_channels, kernel_size=k, padding=p),
        
            nn.BatchNorm2d(output_channels),
        
            nn.PReLU(),
        
            nn.Conv2d(output_channels, output_channels, kernel_size=k, padding=p),
        
            nn.BatchNorm2d(output_channels)
        
        )
        
    def forward(self, x):
        
        output = self.RBnet(x)
        
        return x + ouput
```

# Also we need the `UpSampling Block`

- https://www.cnblogs.com/kk17/p/10094160.html


```python
class UpSamplingBlock(nn.Module):
    def __init__(self, input_channels, scale_factor, k=3, p=1):
        super(UpSamplingBlock, self).__init__()
        
        self.USnet = nn.Sequential(
            
            nn.Conv2d(input_channels, input_channels * (scaleFactor ** 2), kernel_size=k, padding=p),
            
            nn.PixelShuffle(),
            
            nn.PReLU()
            
        )
        
    def forward(self, x):
        output = self.USnet(x)
        
        return output
```

# SRGAN model defination


```python
class Generator(nn.Module):
    def __init__(self, n_residual=8, scale_factor = 2):
        super(Generator, self).__init__()
        self.n_residual = n_residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        for i in range(n_residual):
            self.add_module('residual' + str(i+1), ResidualBlock(64, 64))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        
        self.upsample = nn.Sequential(
            UpsampleBLock(64, scale_factor),
            UpsampleBLock(64, scale_factor),
            nn.Conv2d(64, 3, kernel_size=9, padding=4)
        )

    def forward(self, x):

        y = self.conv1(x)
        cache = y.clone()
        
        for i in range(self.n_residual):
            y = self.__getattr__('residual' + str(i+1))(y)
            
        y = self.conv2(y)
        y = self.upsample(y + cache)

        return (torch.tanh(y) + 1.0) / 2.0
```


```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return F.sigmoid(self.net(x).view(batch_size))
```
