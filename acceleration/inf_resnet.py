from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import math


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)


class InfBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        comp: float = 1.,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = None #conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = None #norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = None #conv3x3(planes, planes, groups=groups)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def load_from_pruned_checkpoints(self, block):
        pconv1 = block.conv1
        pbn1 = block.bn1
        pconv2 = block.conv2
        pbn2 = block.bn2

        ch01 = pconv1.weight.ind
        ch12 = pconv2.weight.ind
        gr1 = pconv1.weight.groups
        gr2 = pconv2.weight.groups
        
        ########################################################################
        # This part is manual implementation of the unused filter removal,     #
        # which is automatically done by compiling the model using torch.jit.  #
        ########################################################################
        
        # removed = (1-ch12).prod(dim=0).squeeze().nonzero(as_tuple=True)[0].tolist()
        # grouplist = [[] for _ in gr1]
        
        # for g, gl in zip(gr1, grouplist):
        #     for r in removed:
        #         if r in g:
        #             gl.append(r)
        
        # num_removed=min([len(gl) for gl in grouplist])
        # remove_list = [gl[:num_removed] for gl in grouplist]
        # print(remove_list)
        # new_gr1 = []
        # for r, g in zip(remove_list, gr1):
        #     list_g = g.tolist()
        #     for f in r:
        #         list_g.remove(f)
        #     new_gr1.append(torch.tensor(list_g, dtype=g.dtype, device=g.device))
        # gr1 = new_gr1
        
        inp1 = ch01[0].count_nonzero().item()
        g1 = len(gr1)
        p1 = pconv1.weight.size(0)#-num_removed*g1
        print(inp1, p1, g1)
        
        inp2 = ch12[0].count_nonzero().item()
        p2 = pconv2.weight.size(0)
        g2 = len(gr2)
        print(inp2, p2, g2)
        
        self.conv1 = conv3x3(inp1*g1, p1, self.stride, groups=g1).to(self.bn2.weight.device)
        self.bn1 = nn.BatchNorm2d(p1).to(self.bn2.weight.device)
        self.conv2 = conv3x3(inp2*g2, p2, groups=g2).to(self.bn2.weight.device)

        sconv1 = self.conv1
        sbn1 = self.bn1
        sconv2 = self.conv2
        sbn2 = self.bn2
        
        group_filters = []
        group_bn_means = []
        group_bn_vars = []
        group_bn_weights = []
        group_bn_bias = []
        
        shuffle_map1 = {}
        sample1 = []
        for g in gr1:
            remaining_ch = ch01[g[0]].view(-1).nonzero().view(-1)
            w = pconv1.weight.data[g][:,remaining_ch]
            bn_m = pbn1.running_mean[g]
            bn_v = pbn1.running_var[g]
            bn_w = pbn1.weight[g]
            bn_b = pbn1.bias[g]
            group_filters.append(w)
            group_bn_means.append(bn_m)
            group_bn_vars.append(bn_v)
            group_bn_weights.append(bn_w)
            group_bn_bias.append(bn_b)
            sample1.append(remaining_ch)

            for ch_ind in g.tolist():
                shuffle_map1[ch_ind] = len(shuffle_map1)
        new_w = torch.cat(group_filters, dim=0)
        new_bn_m = torch.cat(group_bn_means, dim=0)
        new_bn_v = torch.cat(group_bn_vars, dim=0)
        new_bn_w = torch.cat(group_bn_weights, dim=0)
        new_bn_b = torch.cat(group_bn_bias, dim=0)
        self.sample1 = torch.cat(sample1, dim=0)
        sconv1.weight.data.copy_(new_w)
        sbn1.running_mean.data.copy_(new_bn_m)
        sbn1.running_var.data.copy_(new_bn_v)
        sbn1.weight.data.copy_(new_bn_w)
        sbn1.bias.data.copy_(new_bn_b)
        
        group_filters = []
        group_bn_means = []
        group_bn_vars = []
        group_bn_weights = []
        group_bn_bias = []
        
        shuffle_map2 = {}
        sample2 = []
        for g in gr2:
            remaining_ch = ch12[g[0]].view(-1).nonzero().view(-1)
            w = pconv2.weight.data[g][:,remaining_ch]
            bn_m = pbn2.running_mean[g]
            bn_v = pbn2.running_var[g]
            bn_w = pbn2.weight[g]
            bn_b = pbn2.bias[g]
            group_filters.append(w)
            group_bn_means.append(bn_m)
            group_bn_vars.append(bn_v)
            group_bn_weights.append(bn_w)
            group_bn_bias.append(bn_b)
            for r in remaining_ch.tolist():
                sample2.append(shuffle_map1[r])

            for ch_ind in g.tolist():
                shuffle_map2[ch_ind] = len(shuffle_map2)
            
        new_w = torch.cat(group_filters, dim=0)
        new_bn_m = torch.cat(group_bn_means, dim=0)
        new_bn_v = torch.cat(group_bn_vars, dim=0)
        new_bn_w = torch.cat(group_bn_weights, dim=0)
        new_bn_b = torch.cat(group_bn_bias, dim=0)
        self.sample2 = torch.tensor(sample2, dtype=w.dtype).long()
        sconv2.weight.data.copy_(new_w)
        sbn2.running_mean.data.copy_(new_bn_m)
        sbn2.running_var.data.copy_(new_bn_v)
        sbn2.weight.data.copy_(new_bn_w)
        sbn2.bias.data.copy_(new_bn_b)
        self.add_ind = torch.cat(gr2, dim=0)

        if self.downsample is not None:
            self.downsample[0].weight.data.copy_(block.downsample[0].weight.data)
            self.downsample[1].weight.data.copy_(block.downsample[1].weight.data)
            self.downsample[1].bias.data.copy_(block.downsample[1].bias.data)
            self.downsample[1].running_mean.data.copy_(block.downsample[1].running_mean.data)
            self.downsample[1].running_var.data.copy_(block.downsample[1].running_var.data)
            
        

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x[:,self.sample1])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out[:,self.sample2])
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        identity.index_add_(1, self.add_ind, out)
        out = self.relu(identity)

        return out


class InfBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        comp: float = 1.,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #width = int(planes * (base_width / 64.0))
        self.conv1 = None #conv1x1(math.ceil(inplanes*comp)*groups, width, groups=groups)
        self.bn1 = None #norm_layer(width)
        self.conv2 = None #conv3x3(math.ceil(width*comp)*groups, width, stride, groups, dilation)
        self.bn2 = None #norm_layer(width)
        self.conv3 = None #conv1x1(math.ceil(width*comp)*groups, planes * self.expansion, groups=groups)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def load_from_pruned_checkpoints(self, block):
        pconv1 = block.conv1
        pbn1 = block.bn1
        pconv2 = block.conv2
        pbn2 = block.bn2
        pconv3 = block.conv3
        pbn3 = block.bn3

        ch01 = pconv1.weight.ind
        ch12 = pconv2.weight.ind
        ch23 = pconv3.weight.ind
        gr1 = pconv1.weight.groups
        gr2 = pconv2.weight.groups
        gr3 = pconv3.weight.groups
        
        
        inp1 = ch01[0].count_nonzero().item()
        g1 = len(gr1)
        p1 = pconv1.weight.size(0)#-num_removed*g1
        print(inp1, p1, g1)
        
        inp2 = ch12[0].count_nonzero().item()
        p2 = pconv2.weight.size(0)
        g2 = len(gr2)
        print(inp2, p2, g2)
        
        inp3 = ch23[0].count_nonzero().item()
        p3 = pconv3.weight.size(0)
        g3 = len(gr3)
        print(inp3, p3, g3)
        
        self.conv1 = conv1x1(inp1*g1, p1, groups=g1).to(self.bn3.weight.device)
        self.bn1 = nn.BatchNorm2d(p1).to(self.bn3.weight.device)
        self.conv2 = conv3x3(inp2*g2, p2, self.stride, groups=g2).to(self.bn3.weight.device)
        self.bn2 = nn.BatchNorm2d(p2).to(self.bn3.weight.device)
        self.conv3 = conv1x1(inp3*g3, p3, groups=g3).to(self.bn3.weight.device)
        
        
        sconv1 = self.conv1
        sbn1 = self.bn1
        sconv2 = self.conv2
        sbn2 = self.bn2
        sconv3 = self.conv3
        sbn3 = self.bn3
        
        
        group_filters = []
        group_bn_means = []
        group_bn_vars = []
        group_bn_weights = []
        group_bn_bias = []
        
        shuffle_map1 = {}
        sample1 = []
        for g in gr1:
            remaining_ch = ch01[g[0]].view(-1).nonzero().view(-1)
            w = pconv1.weight.data[g][:,remaining_ch]
            bn_m = pbn1.running_mean[g]
            bn_v = pbn1.running_var[g]
            bn_w = pbn1.weight[g]
            bn_b = pbn1.bias[g]
            group_filters.append(w)
            group_bn_means.append(bn_m)
            group_bn_vars.append(bn_v)
            group_bn_weights.append(bn_w)
            group_bn_bias.append(bn_b)
            sample1.append(remaining_ch)

            for ch_ind in g.tolist():
                shuffle_map1[ch_ind] = len(shuffle_map1)
            
        new_w = torch.cat(group_filters, dim=0)
        new_bn_m = torch.cat(group_bn_means, dim=0)
        new_bn_v = torch.cat(group_bn_vars, dim=0)
        new_bn_w = torch.cat(group_bn_weights, dim=0)
        new_bn_b = torch.cat(group_bn_bias, dim=0)
        self.sample1 = torch.cat(sample1, dim=0)
        sconv1.weight.data.copy_(new_w)
        sbn1.running_mean.data.copy_(new_bn_m)
        sbn1.running_var.data.copy_(new_bn_v)
        sbn1.weight.data.copy_(new_bn_w)
        sbn1.bias.data.copy_(new_bn_b)
        
        group_filters = []
        group_bn_means = []
        group_bn_vars = []
        group_bn_weights = []
        group_bn_bias = []
        
        shuffle_map2 = {}
        sample2 = []
        for g in gr2:
            remaining_ch = ch12[g[0]].view(-1).nonzero().view(-1)
            w = pconv2.weight.data[g][:,remaining_ch]
            bn_m = pbn2.running_mean[g]
            bn_v = pbn2.running_var[g]
            bn_w = pbn2.weight[g]
            bn_b = pbn2.bias[g]
            group_filters.append(w)
            group_bn_means.append(bn_m)
            group_bn_vars.append(bn_v)
            group_bn_weights.append(bn_w)
            group_bn_bias.append(bn_b)
            sample2.append([shuffle_map1[r] for r in remaining_ch.tolist()])

            for ch_ind in g.tolist():
                shuffle_map2[ch_ind] = len(shuffle_map2)
        ch_size = min([g.size(1) for g in group_filters])
        group_filters = [g[:,:ch_size,:,:] for g in group_filters]
        sample2 = [g[:ch_size] for g in sample2]
            
        new_w = torch.cat(group_filters, dim=0)
        new_bn_m = torch.cat(group_bn_means, dim=0)
        new_bn_v = torch.cat(group_bn_vars, dim=0)
        new_bn_w = torch.cat(group_bn_weights, dim=0)
        new_bn_b = torch.cat(group_bn_bias, dim=0)
        self.sample2 = torch.tensor(sample2, dtype=w.dtype).view(-1).long()
        sconv2.weight.data.copy_(new_w)
        sbn2.running_mean.data.copy_(new_bn_m)
        sbn2.running_var.data.copy_(new_bn_v)
        sbn2.weight.data.copy_(new_bn_w)
        sbn2.bias.data.copy_(new_bn_b)

        group_filters = []
        group_bn_means = []
        group_bn_vars = []
        group_bn_weights = []
        group_bn_bias = []
        
        shuffle_map3 = {}
        sample3 = []
        for g in gr3:
            remaining_ch = ch23[g[0]].view(-1).nonzero().view(-1)
            w = pconv3.weight.data[g][:,remaining_ch]
            bn_m = pbn3.running_mean[g]
            bn_v = pbn3.running_var[g]
            bn_w = pbn3.weight[g]
            bn_b = pbn3.bias[g]
            group_filters.append(w)
            group_bn_means.append(bn_m)
            group_bn_vars.append(bn_v)
            group_bn_weights.append(bn_w)
            group_bn_bias.append(bn_b)
            for r in remaining_ch.tolist():
                sample3.append(shuffle_map2[r])

            for ch_ind in g.tolist():
                shuffle_map3[ch_ind] = len(shuffle_map3)
            
        new_w = torch.cat(group_filters, dim=0)
        new_bn_m = torch.cat(group_bn_means, dim=0)
        new_bn_v = torch.cat(group_bn_vars, dim=0)
        new_bn_w = torch.cat(group_bn_weights, dim=0)
        new_bn_b = torch.cat(group_bn_bias, dim=0)
        self.sample3 = torch.tensor(sample3, dtype=w.dtype).long()
        sconv3.weight.data.copy_(new_w)
        sbn3.running_mean.data.copy_(new_bn_m)
        sbn3.running_var.data.copy_(new_bn_v)
        sbn3.weight.data.copy_(new_bn_w)
        sbn3.bias.data.copy_(new_bn_b)

        self.add_ind = torch.cat(gr3, dim=0)

        if self.downsample is not None:
            self.downsample[0].weight.data.copy_(block.downsample[0].weight.data)
            self.downsample[1].weight.data.copy_(block.downsample[1].weight.data)
            self.downsample[1].bias.data.copy_(block.downsample[1].bias.data)
            self.downsample[1].running_mean.data.copy_(block.downsample[1].running_mean.data)
            self.downsample[1].running_var.data.copy_(block.downsample[1].running_var.data)
            

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x[:,self.sample1])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out[:,self.sample2])
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out[:,self.sample3])
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        identity.index_add_(1, self.add_ind, out)
        out = self.relu(identity)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[InfBasicBlock, InfBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, InfBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, InfBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[InfBasicBlock, InfBottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def load_from_pruned_checkpoints(self, block):
        self.conv1.weight.data.copy_(block.conv1.weight.data)
        self.bn1.weight.data.copy_(block.bn1.weight.data)
        self.bn1.bias.data.copy_(block.bn1.bias.data)
        self.bn1.running_mean.data.copy_(block.bn1.running_mean.data)
        self.bn1.running_var.data.copy_(block.bn1.running_var.data)
        self.fc.weight.data.copy_(block.fc.weight.data)
        self.fc.bias.data.copy_(block.fc.bias.data)
            
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[InfBasicBlock, InfBottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def inf_resnet18(**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(InfBasicBlock, [2, 2, 2, 2], **kwargs)


def inf_resnet50(**kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(InfBottleneck, [3, 4, 6, 3], **kwargs)

