from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu, sigmoid



class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        super(UNet, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )


    def forward(self, x):
        return self.double_conv(x)
    


if __name__ == "__main__":
    from data_e import FootballDataset
    
    dataset = FootballDataset()
    model = UNet(3, 1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    EPOCH = 15


    for images, masks in dataloader:
        pass

        
    
