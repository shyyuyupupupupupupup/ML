import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self,dim):
        super(ResNetBlock,self).__init__()
        self.resblock = self.build_conv_block(dim)

    def build_conv_block(self,dim):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim,dim,kernel_size=3, padding=1)]
        return nn.Sequential(*conv_block)