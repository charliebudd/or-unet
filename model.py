from typing import List
import torch
from torch.nn import Module, ModuleList, Conv2d, BatchNorm2d, ReLU, ConvTranspose2d

class OrUnet(Module):

    def __init__(self, input_channels=3, output_classes=2, features=[48, 96, 192, 384, 512, 512, 512], extra_res_blocks=[1, 2, 3, 4, 4, 4], output_count=5):
        super().__init__()

        self.in_block = InBlock(input_channels, features[0])

        in_features, out_features = features[:-1], features[1:]
        self.down_blocks = ModuleList([DownBlock(in_f, out_f, n_res) for in_f, out_f, n_res in zip(in_features, out_features, extra_res_blocks)])

        in_features, out_features = out_features[::-1], in_features[::-1]
        self.up_blocks =  ModuleList([UpBlock(in_f, out_f) for in_f, out_f in zip(in_features, out_features)])

        self.output_count = output_count
        self.output_blocks =  ModuleList([OutBlock(in_f, output_classes) for in_f in features[:output_count]])

    def forward(self, x):

        x = self.in_block(x)

        x, skips = self.encoder(x)

        if self.training:
           output = self.training_decoder(x, skips)
        else:
           output = self.evaluation_decoder(x, skips)
        
        return output

    def encoder(self, x):
        skips = []
        for down_block in self.down_blocks:
            skips.append(x)
            x = down_block(x)
        skips.reverse()
        return x, skips

    def training_decoder(self, x, skips: List[torch.Tensor]):
        outputs = []
        for index, up_block in enumerate(self.up_blocks):
            x = up_block(x, skips[index])
            if len(self.up_blocks) - index <= self.output_count:
                outputs.append(x)
        outputs.reverse()

        for index, out_block in enumerate(self.output_blocks):
            outputs[index] = out_block(outputs[index])

        return outputs

    def evaluation_decoder(self, x, skips: List[torch.Tensor]):
        for index, up_block in enumerate(self.up_blocks):
            x = up_block(x, skips[index])
        return [self.output_blocks[0](x)]

class InBlock(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.cbr = ConvBnReLU(in_features, out_features)
        self.res = ResBlock(out_features, out_features)

    def forward(self, x):
        x = self.cbr(x)
        x = self.res(x)
        return x

class DownBlock(Module):
    def __init__(self, in_features, out_features, extra_res_count):
        super().__init__()
        self.down_res = ResBlock(in_features, out_features, down_sample=True)
        self.extra_res =  ModuleList([ResBlock(out_features, out_features) for _ in range(extra_res_count)])

    def forward(self, x):
        x = self.down_res(x)
        for res in self.extra_res:
            x = res(x)
        return x

class UpBlock(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.upsample = ConvTranspose2d(in_features, out_features, kernel_size=2, stride=2, bias=False)
        self.cbr1 = ConvBnReLU(2 * out_features, out_features)
        self.cbr2 = ConvBnReLU(out_features, out_features)

    def forward(self, x, y):
        x = self.upsample(x)
        x = torch.cat([x, y], dim=1)
        x = self.cbr1(x)
        x = self.cbr2(x)
        return x

class OutBlock(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = Conv2d(in_features, out_features, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class ResBlock(Module):
    def __init__(self, in_features, out_features, down_sample=False):
        super().__init__()
        stride = 2 if down_sample else 1
        self.skip_conv = Conv2d(in_features, out_features, stride=stride, kernel_size=1, bias=False)
        self.brc1 = BnReLUConv(in_features, out_features, stride)
        self.brc2 = BnReLUConv(out_features, out_features)
        
    def forward(self, x):
        skip = self.skip_conv(x)
        x = self.brc1(x)
        x = self.brc2(x)
        x += skip
        return x

class ConvBnReLU(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = Conv2d(in_features, out_features, kernel_size=3, padding=1, bias=False)
        self.batch_norm = BatchNorm2d(out_features)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class BnReLUConv(Module):
    def __init__(self, in_features, out_features, stride=1):
        super().__init__()
        self.batch_norm = BatchNorm2d(in_features)
        self.relu = ReLU(inplace=True)
        self.conv = Conv2d(in_features, out_features, stride=stride, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
        