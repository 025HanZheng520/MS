import torch
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn.functional as F
import math
from functools import partial
from torch.nn.modules.utils import _pair
import random
torch.set_printoptions(precision=3,edgeitems=32,linewidth=350)
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs)[0] + x,self.fn(x, **kwargs)[1]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x),x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask


        attn = torch.softmax(dots,dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out,attn[:,0,0]*attn[:,1,0]*attn[:,2,0]*attn[:,3,0]##visualization


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        b,n,_=x.shape
        map=torch.ones(b,n).cuda()
        i=0
        for attn, ff in self.layers:
            x,attn_map = attn(x, mask=mask)
            x,_ = ff(x)
            # if map[0,0]==1:
            # if i>=1:
            map=map*attn_map
            # i=i+1
        return x

class PatchEmbed(nn.Module):
    def __init__(self,
                 img_size,
                 patch_size,
                 inplanes):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_sizes = [_pair(p) for p in patch_size]  # 修正这里的属性名为 patch_sizes

        self.num_patches = [
            (self.img_size[1] // ps[1]) * (self.img_size[0] // ps[0])
            for ps in self.patch_sizes
        ] #[14*14, 7*7 ]
        #print(self.num_patches)

        # 使用卷积层进行嵌入
        self.patch_embedding = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inplanes, inplanes, kernel_size=ps, stride=ps),
                nn.BatchNorm2d(inplanes),
                nn.ReLU()
            ) for ps in self.patch_sizes
        ])

    def forward(self, x):
        embedded_patches = []
        for layer in self.patch_embedding:
            x1 = layer(x)
            embedded_patches.append(x1)
        return embedded_patches

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

##building block of ResNet18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.attn = nn.Sequential(
        nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=False),  # 32*33*33
        nn.BatchNorm2d(1),
        nn.Sigmoid(),
        )
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, img_size = 28,replace_stride_with_dilation=None, norm_layer=None,
                 num_patches=7*7,  base_dim=128,depth=2, heads=4, mlp_dim=512, dim_head=32, dropout=0.):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.img_size = img_size
        self.patch_size = [1, 2, 4]

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or"
                             " a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(self.inplanes),
                                   nn.ReLU(),
                                   nn.Conv2d(self.inplanes, self.inplanes,kernel_size=3,stride=1,padding=1,bias=False))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ## Conv1-3 layer
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        ## the dynamic branch of the DSF module
        self.d_branch = self._make_layer(block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.patch_embedding=nn.Sequential(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1,bias=False),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(),
                                           )


        self.cls_token = nn.Parameter(torch.randn(1, 1, 256*4*4))
        #self.temporal_pos_embedding = nn.Parameter(torch.randn(1, 16, dim))
        #self.pos_embedding_static = nn.Parameter(torch.randn(1, 16, 256))
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            inplanes=self.inplanes)
        self.pos_embedding = nn.ParameterList([  # (1, num_patches, embed_dims)

            nn.Parameter(torch.randn(1, num_patches, base_dim * 2 ** (i)))
            for i, num_patches in enumerate(self.patch_embed.num_patches)
        ])

        #self.temporal_transformer = Transformer(dim, depth, heads, dim_head, 512, dropout)
        #self.s_branch = Transformer(dim=256, depth=2, heads=4, dim_head=32, mlp_dim=256, dropout=0.)
        self.stransformer = Transformer(base_dim, depth, heads, dim_head, mlp_dim, dropout)
        self.stransformer1 = Transformer(base_dim * 2 ** 1, depth, heads, dim_head, mlp_dim, dropout)
        self.stransformer2 = Transformer(base_dim * 2 ** 2, depth, heads, dim_head, mlp_dim, dropout)
        #self.temporal_transformer = Transformer(base_dim, depth, heads, dim_head, mlp_dim, dropout)
        #self.temporal_transformer1 = Transformer(base_dim * 2 ** 1, depth, heads, dim_head, mlp_dim, dropout)
        #self.temporal_transformer2 = Transformer(base_dim * 2 ** 2, depth, heads, dim_head, mlp_dim, dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = x.contiguous().view(-1, 3, 224,224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        #print(x3.size())
        ##dynamic static fusion module
        # act = ((x.view(b // 16, 16, c, h, w))[:, 1:16, :, :] - (x.view(b // 16, 16, c, h, w))[:, 0:15, :, :]).view(-1, c, h,w)
        mul_shape = [x1, x2, x3]
        # print(str(len(mul_shape))+str(x1.size())+str(x2.size())+str(x3.size()))

        dy_scale = []
        for index, sets in enumerate(mul_shape):
            # print(sets.size())
            b, c, h, w = sets.shape
            dy_data = ((sets.view(b // 16, 16, c, h, w))[:, 1:16, :, :] - (sets.view(b // 16, 16, c, h, w))[:, 0:15, :,
                                                                          :]).view(-1, c, h, w)
            # print(dy_data.size())
            last_data = ((sets.view(b // 16, 16, c, h, w))[:, 0, :, :, :] - (sets.view(b // 16, 16, c, h, w))[:, -1, :,
                                                                            :, :]).view(-1, c, h, w)
            # print(last_data.size())
            dy_data = torch.cat((last_data, dy_data), dim=0)
            dy_scale.append(dy_data)
        # print(dy_scale[0].size(),dy_scale[1].size(),dy_scale[2].size())

        s_result = []
        for index, data in enumerate(mul_shape):
            # print(index, data.size())
            b_l, c, h, w = data.shape
            data = data.reshape((b_l, c, h * w)).permute(0, 2, 1)
            dy_scale[index] = dy_scale[index].reshape((-1, c, h * w)).permute(0, 2, 1)
            b, n, _ = data.shape
            data = data + self.pos_embedding[index][:, :n]
            if index == 0:
                # print(0)
                data = self.stransformer(data)
                dy_scale[index] = self.stransformer(dy_scale[index])
                # print( data.size())
                # fuser
            elif index == 1:
                # print(1)
                data = self.stransformer1(data)
                dy_scale[index] = self.stransformer1(dy_scale[index])
            elif index == 2:
                # print(2)
                data = self.stransformer2(data)
                dy_scale[index] = self.stransformer2(dy_scale[index])
            data = data.permute(0, 2, 1).reshape((b, c, h, w))
            dy_scale[index] = dy_scale[index].permute(0, 2, 1).reshape((b, c, h, w))
            # print("data: "+str(index)+str(data.size()))
            data = self.avgpool(data)
            dy_scale[index] = self.avgpool(dy_scale[index])
            # print("data: " + str(index) + str(data.size()))
            data = torch.flatten(data, 1)
            dy_scale[index] = torch.flatten(dy_scale[index], 1)
            # print("data: " + str(index) + str(data.size()))

            s_result.append(data)
            dy_scale[index] = dy_scale[index]

        for index, data_s in enumerate(s_result):  # 256
            b_l, c = data_s.shape
            data_s = data_s.contiguous().view(-1, 16, c)
            dy_scale[index]= dy_scale[index].contiguous().view(-1, 16, c)
            b, n, _ = data_s.shape
            # cls_tokens = repeat(self.cls_token[0], '() n d -> b n d', b=b)
            cls_tokens = torch.mean(data_s, dim=1).unsqueeze(1)
            dy_tokens = torch.mean(dy_scale[index], dim=1).unsqueeze(1)

            data_s = torch.cat((cls_tokens, data_s), dim=1)
            data_dy = torch.cat((dy_tokens, dy_scale[index]), dim=1)
            if c == 128:
                #print(0)
                data_s = data_s + self.pos_embedding[0][:, :(n + 1)]
                data_s = self.stransformer(data_s)

                data_dy = data_dy+ self.pos_embedding[0][:, :(n + 1)]
                data_dy = self.stransformer(data_dy)
            elif c == 256:
                # print(1)
                data_s = data_s + self.pos_embedding[1][:, :(n + 1)]
                data_s = self.stransformer1(data_s)

                data_dy = data_dy + self.pos_embedding[1][:, :(n + 1)]
                data_dy = self.stransformer1(data_dy)
            elif c == 512:

                data_s = data_s + self.pos_embedding[2][:, :(n + 1)]
                data_s = self.stransformer2(data_s)

                data_dy = data_dy + self.pos_embedding[2][:, :(n + 1)]
                data_dy = self.stransformer2(data_dy)
            s_result[index] = data_s
            dy_scale[index] =data_dy
            #print([s_result[i].size() for i in range(len(s_result))])
            #print([s_result[i].size() for i in range(len(s_result))])

        return s_result,dy_scale


def backbone():
    return ResNet(BasicBlock, [1, 1, 1, 3])


if __name__ == '__main__':
    img = torch.randn((2, 32, 3, 224, 224))
    model = backbone().to('cuda:0')
    model(img)
