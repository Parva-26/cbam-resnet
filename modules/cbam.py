
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):
        
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)  

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.shared_mlp(self.avg_pool(x))   
        max_out = self.shared_mlp(self.max_pool(x))   

        channel_weights = self.sigmoid(avg_out + max_out)  

        return x * channel_weights


class SpatialAttention(nn.Module):
    

    def __init__(self, kernel_size=7):
        
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=2,           
            out_channels=1,          
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  

        pooled = torch.cat([avg_out, max_out], dim=1)

        spatial_weights = self.sigmoid(self.conv(pooled))

        return x * spatial_weights


class CBAM(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x
