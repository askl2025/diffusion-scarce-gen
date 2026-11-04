"""mmcv.cnn shim -- only ConvModule is needed by CLRNet."""

import torch.nn as nn

CONV_LAYERS = {
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
}

NORMS = {
    'BN': nn.BatchNorm2d,
    'BN1d': nn.BatchNorm1d,
    'BN2d': nn.BatchNorm2d,
    'BN3d': nn.BatchNorm3d,
    'SyncBN': nn.SyncBatchNorm,
    'GN': nn.GroupNorm,
    'LN': nn.LayerNorm,
    'IN': nn.InstanceNorm2d,
}

ACTS = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'PReLU': nn.PReLU,
    'RReLU': nn.RReLU,
    'ReLU6': nn.ReLU6,
    'ELU': nn.ELU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'GELU': nn.GELU,
    'Hardswish': nn.Hardswish,
}

INPLACE_ACTS = {'ReLU', 'LeakyReLU', 'RReLU', 'ELU', 'ReLU6', 'Hardswish'}


def _build_conv(conv_cfg, in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, bias):
    if conv_cfg is None:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias)
    cfg = dict(conv_cfg)
    layer_type = cfg.pop('type', 'Conv2d')
    return CONV_LAYERS[layer_type](in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=groups, bias=bias,
                                   **cfg)


def build_norm_layer(norm_cfg, num_features, postfix=''):
    """Return (name, layer). mmcv names the attribute <type>.lower() + postfix,
    e.g. 'BN' -> 'bn', which is what the official checkpoints key on."""
    cfg = dict(norm_cfg)
    layer_type = cfg.pop('type')
    if layer_type == 'GN':
        # mmcv: num_groups defaults to 32
        cfg.setdefault('num_groups', 32)
        cfg.setdefault('num_channels', num_features)
        layer = nn.GroupNorm(**cfg)
    else:
        layer = NORMS[layer_type](num_features, **cfg)
    return layer_type.lower() + postfix, layer


def build_activation_layer(act_cfg, inplace):
    cfg = dict(act_cfg)
    layer_type = cfg.pop('type')
    if layer_type in INPLACE_ACTS:
        cfg.setdefault('inplace', inplace)
    return ACTS[layer_type](**cfg)


class ConvModule(nn.Module):
    """conv -> (norm) -> (act), matching mmcv's default ordering.

    Mirrors upstream's `bias='auto'` rule: a normalisation layer implies
    bias=False, otherwise bias=True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 order=('conv', 'norm', 'act')):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.order = order
        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if bias == 'auto':
            bias = False if norm_cfg is not None else True
        self.with_norm = norm_cfg is not None
        self.with_act = act_cfg is not None

        self.conv = _build_conv(conv_cfg, in_channels, out_channels,
                                kernel_size, stride, padding, dilation, groups,
                                bias)

        # mmcv stores the norm under its own name ('bn', 'gn', ...), not
        # 'norm'; the official CLRNet checkpoint keys depend on this.
        if self.with_norm:
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            setattr(self, self.norm_name, norm)
        if self.with_act:
            self.activate = build_activation_layer(act_cfg, inplace)

        self.init_weights()

    def init_weights(self):
        # upstream applies kaiming for conv and constant for norm; the
        # checkpoint overwrites every parameter anyway.
        if self.with_norm:
            norm = getattr(self, self.norm_name)
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)

    def forward(self, x):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.with_norm:
                x = getattr(self, self.norm_name)(x)
            elif layer == 'act' and self.with_act:
                x = self.activate(x)
        return x

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.conv}, norm={self.with_norm}, '
                f'act={self.with_act})')
