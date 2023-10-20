from typing import Callable, Any, Optional, List
import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, neuron
from spiking_morph import ErosionLayer, DilationLayer
from collections import OrderedDict

class DepthSepConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer: callable,
                 bias: bool,
                 ksize=3,
                 stride=1,
                 padding=1,
                 multiplier=1, 
                 neuron: callable = None,
                 dw=True,
                 **kwargs
                ):
        super().__init__()
        self.dw = dw
        
        if self.dw:
            self.pad = nn.ConstantPad2d(padding, 0.)
            self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=in_channels,
                                         kernel_size=ksize,
                                         stride=stride,
                                         padding=0,
                                         groups=in_channels, bias=bias)
            self.ero0 = ErosionLayer()
            self.dil0 = DilationLayer()
            self.bn1 = norm_layer(in_channels)

            self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         groups=1, bias=bias)
            self.ero1 = ErosionLayer()
            self.dil1 = DilationLayer()
            self.bn2 = norm_layer(in_channels)
        else:
            self.pad = nn.ConstantPad2d(padding, 0.)
            self.conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=ksize,
                                         stride=stride,
                                         padding=0, bias=bias)
            self.ero0 = ErosionLayer()
            self.dil0 = DilationLayer()
            self.bn = norm_layer(in_channels)
        
        self.act = neuron(**kwargs)

    def forward(self, x):
        if self.dw:
            x = self.depthwise_conv(self.bn1(self.pad(x)))
            x = self.pointwise_conv(self.bn2(x))
            x = self.act(x)
        else:
            x = self.act(self.conv(self.bn(self.pad(x))))
        return x



class SpikingMobileNet(nn.Module):
    def __init__(self, num_init_channels=2, in_channels=64, num_classes: int = 2, multiplier=1, init_weights=True, norm_layer: callable = None, neuron: callable = None, dw=True, **kwargs):
        super().__init__()
        
        self.nz, self.numel = {}, {}
        self.out_channels = []
        
        if norm_layer is None:
            norm_layer = nn.Identity
        bias = isinstance(norm_layer, nn.Identity)
            
        self.features = nn.Sequential(
            nn.Sequential(OrderedDict(
                [
                    ("pad0", nn.ConstantPad2d(1, 0.)),
                    ("norm0", norm_layer(num_init_channels)),
                    ("conv0", nn.Conv2d(num_init_channels, in_channels, 
                                        kernel_size=3, stride=2, padding=0, bias=bias)),
                    ("ero0", ErosionLayer()),
                    ("dil0", DilationLayer()),
                    ("act0", neuron(**kwargs)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )),
            DepthSepConv(in_channels, in_channels*2, norm_layer, bias, stride=1, neuron=neuron, dw=dw, **kwargs),
            DepthSepConv(in_channels*2, in_channels*2, norm_layer, bias, stride=1, neuron=neuron, dw=dw, **kwargs),
            DepthSepConv(in_channels*2, in_channels*4, norm_layer, bias, stride=2, neuron=neuron, dw=dw, **kwargs),
            DepthSepConv(in_channels*4, in_channels*4, norm_layer, bias, stride=1, neuron=neuron, dw=dw, **kwargs),
            DepthSepConv(in_channels*4, in_channels*8, norm_layer, bias, stride=2, neuron=neuron, dw=dw, **kwargs),
            DepthSepConv(in_channels*8, in_channels*8, norm_layer, bias, stride=1, neuron=neuron, dw=dw, **kwargs),
            DepthSepConv(in_channels*8, in_channels*16, norm_layer, bias, stride=2, neuron=neuron, dw=dw, **kwargs),
            DepthSepConv(in_channels*16, in_channels*16, norm_layer, bias, stride=1, neuron=neuron, dw=dw, **kwargs)
        )

        self.out_channels = [in_channels*2, in_channels*8, in_channels*16]
    
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("norm_classif", norm_layer(in_channels*16)),
                    ("conv_classif", nn.Conv2d(in_channels*16, num_classes, 
                                                kernel_size=1, bias=bias)),
                    ("act_classif", neuron(**kwargs)),
                ]
            )
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        self.reset_nz_numel()
        x = self.features(x)
        x = self.classifier(x)
        x = x.flatten(start_dim=-2).sum(dim=-1)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def add_hooks(self):
        def get_nz(name):
            def hook(model, input, output):
                self.nz[name] += torch.count_nonzero(output)
                self.numel[name] += output.numel()
            return hook
        
        self.hooks = {}
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0
            self.hooks[name] = module.register_forward_hook(get_nz(name))
                
    def reset_nz_numel(self):
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0
        
    def get_nz_numel(self):
        return self.nz, self.numel
    
def sequential_forward(sequential, x_seq):
    assert isinstance(sequential, nn.Sequential)
    out = x_seq
    for i in range(len(sequential)):
        m = sequential[i]
        if isinstance(m, neuron.BaseNode):
            out = m(out)
        else:
            out = functional.seq_to_ann_forward(out, m)
    return out


def spiking_mobilenet(num_init_channels, **kwargs: Any):
    r"""A spiking version of MobileNet V1 model from 
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    Args:
        num_init_channels (int): number of channels of the input data
    """
    return SpikingMobileNet(num_init_channels, **kwargs)