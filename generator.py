import torch, timm
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid_channels = out_channels * 4

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.PixelShuffle(2)
        )
    
    def forward(self, x):
        return self.block(x)
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)

class ViTGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.encoder.head = nn.Identity()
        self.decoderBlock1 = DecoderBlock(768, 256)
        self.decoderBlock2 = DecoderBlock(256, 128)
        self.decoderBlock3 = DecoderBlock(128, 64)
        self.decoderBlock4 = DecoderBlock(64, 32)

        self.bottleneck1 = BottleneckBlock(1024, 256)
        self.bottleneck2 = BottleneckBlock(896, 128)
        self.bottleneck3 = BottleneckBlock(832, 64)

        self.final_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def skip_connection(self, skip_tensor, target_size):
        x = skip_tensor[:, 1:, :]
        x = x.reshape(-1, 14, 14, 768)
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        return x
    
    def forward(self, x):
        skips = self.encoder.get_intermediate_layers(x, n=(3, 6, 9, 11))
        x = self.skip_connection(skips[3], 14)
        
        x = self.decoderBlock1(x)
        res3 = self.skip_connection(skips[2], 28)
        x = torch.cat([x, res3], dim=1)
        x = self.bottleneck1(x)

        x = self.decoderBlock2(x)
        res2 = self.skip_connection(skips[1], 56)
        x = torch.cat([x, res2], dim=1)
        x = self.bottleneck2(x)

        x = self.decoderBlock3(x)
        res1 = self.skip_connection(skips[0], 112)
        x = torch.cat([x, res1], dim=1)
        x = self.bottleneck3(x)

        x = self.decoderBlock4(x)
        x = self.final_head(x)
        return x