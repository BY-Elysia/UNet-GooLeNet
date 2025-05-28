import torch.nn as nn
import torch
import torch.nn.functional as F


from nets.deform_conv_v2 import DeformConv2d
from nets.tasks import TransformerDecoder
from  torchsummary import summary

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        # if(out.shape[0] == 1):
        #     print(1)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)

        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class CoordAtt(nn.Module):
    def __init__(self, inp1,inp2, oup, reduction=4):
        super(CoordAtt, self).__init__()
        mip = inp1 // reduction
        self.conv1 = nn.Conv2d(inp1, inp1 // reduction, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inp1 // reduction)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(inp2, inp1 // reduction, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(inp1 // reduction)
        self.relu2 = nn.ReLU()
        self.conv_d = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_x = nn.Conv2d(inp2, inp1, kernel_size=1, stride=1, padding=0)
    def forward(self, g, x):
        b, c, h, w = x.size()

        g_h = F.adaptive_avg_pool2d(g, (h, 1))
        g_w = F.adaptive_avg_pool2d(g, (1, w)).permute(0, 1,3,2)

        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1,3,2)

        g_y = torch.cat([g_h, g_w], dim=2)
        g_y = self.conv1(g_y)
        g_y = self.bn1(g_y)
        g_y = self.relu1(g_y)

        x_y = torch.cat([x_h, x_w], dim=2)
        x_y = self.conv2(x_y)
        x_y = self.bn2(x_y)
        x_y = self.relu2(x_y)

        g_h, g_w = torch.split(g_y, [h, w], dim=2)
        g_w = g_w.permute(0, 1, 3,2)
        x_h, x_w = torch.split(x_y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = (x_h + g_h) / 2
        a_w = (x_w + g_w) / 2

        a_h, a_w =  torch.sigmoid(self.conv_h(a_h)), torch.sigmoid(self.conv_w(a_w))
        # x= self.conv_x(x)
        x = x * a_h * a_w
        return x
class UpBlockAlig(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlockAlig, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, (2, 2), 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.cca = CoordAtt3(in_channels//2)
    def forward(self, x, skip_x):
        out = self.up(x)
        skip_x = self.cca(skip_x, out)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)
        # return self.nConvs(skip_x)



class UNetBasic(nn.Module):
    def __init__(self, n_channels=3, n_classes=9):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1))

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpoll1 = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpoll2 = nn.AdaptiveMaxPool2d((1, 1))

        self.fc1 = nn.Linear(in_channels*8,in_channels*4)
        self.fc2 = nn.Linear(in_channels*4,1)

        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # Question here
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.down3(x3)

        # cl_out1_a = self.avgpool1(x4)
        # cl_out1_m = self.maxpoll1(x4)
        x5 = self.down4(x4)

        # cl_out_2_cl, x5 = self.task2(x5, x5)
        cl_out2_a = self.avgpool2(x5)
        # cl_out2_m = self.maxpoll2(x5)


        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        # cl_out = torch.flatten(torch.cat([cl_out1_a, cl_out1_m, cl_out2_a, cl_out2_m], dim=1), 1)
        cl_out = torch.flatten( cl_out2_a,1)

        cl_out = self.fc1(cl_out)
        cl_out = self.fc2(cl_out)


        logits = self.outc(x)
        return logits, cl_out
class CoordAtt3(nn.Module):
    def __init__(self, inp):
        super(CoordAtt3, self).__init__()
        self.conv1_e = _make_nConv(inp, inp, 1, 'ReLU')
        self.conv2_e = _make_nConv(inp, inp, 1, 'ReLU')
        self.avgpool_e = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool_e = nn.AdaptiveMaxPool2d((1,1))
        # self.soft_pool_e = AdaptiveSoftPool2d(output_size=(1, 1))
        self.fc_avg = nn.Conv2d(inp,inp//2,kernel_size=1, stride=1, padding=0)
        self.fc_max = nn.Conv2d(inp,inp//2,kernel_size=1, stride=1, padding=0)
        self.fc_soft = nn.Conv2d(inp,inp//2,kernel_size=1, stride=1, padding=0)
        self.fc_avg_max_sfot = nn.Conv2d(inp//2,inp,kernel_size=1, stride=1, padding=0)
        self.deformabel = DeformConv2d(in_channels=inp, out_channels=inp,kernel_size=3)

    def forward(self, e, d):
        e_1 = self.conv1_e(e)
        avg = self.avgpool_e(e_1)
        max = self.maxpool_e(e_1)

        fc_avg = self.fc_avg(avg)
        fc_max = self.fc_max(max)

        avg_max = F.relu(fc_avg)+F.relu(fc_max)

        avg_max_soft = F.sigmoid(self.fc_avg_max_sfot(avg_max))

        def_d = self.conv2_e(d)
        avg_max_soft_def = avg_max_soft*def_d
        out = e_1+ avg_max_soft_def+def_d

        return out





class UNetTask(nn.Module):
    def __init__(self, n_channels=3, n_classes=9):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1))

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpoll1 = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpoll2 = nn.AdaptiveMaxPool2d((1, 1))
        self.task2 = TransformerDecoder(dim=in_channels * 8, depth=1, heads=8, dim_head=64,
                                        mlp_dim=2048, dropout=0,
                                        decoder_pos_size=14, softmax=True)

        self.fc1 = nn.Linear(in_channels * 8, in_channels * 4)
        self.fc2 = nn.Linear(in_channels * 4, 1)
        self.loss_function = AutomaticWeightedLoss(num=2)
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        deeps =[]
        # Question here
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.down3(x3)

        # cl_out1_a = self.avgpool1(x4)
        # cl_out1_m = self.maxpoll1(x4)
        x5 = self.down4(x4)

        cl_out_2_cl, x5 = self.task2(x5, x5)
        cl_out2_a = self.avgpool2(cl_out_2_cl)
        # cl_out2_m = self.maxpoll2(cl_out_2_cl)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        # cl_out = torch.flatten(torch.cat([cl_out1_a, cl_out1_m, cl_out2_a, cl_out2_m], dim=1), 1)
        cl_out = torch.flatten(cl_out2_a , 1)
        cl_out = self.fc1(cl_out)
        cl_out = self.fc2(cl_out)
        logits = self.outc(x)
        return logits, cl_out
class UNetTaskAlig(nn.Module):
    def __init__(self, n_channels=3, n_classes=9):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Question here
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpBlockAlig(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlockAlig(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlockAlig(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpBlockAlig(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1))

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpoll1 = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpoll2 = nn.AdaptiveMaxPool2d((1, 1))
        self.task2 = TransformerDecoder(dim=in_channels * 8, depth=1, heads=8, dim_head=64,
                                        mlp_dim=2048, dropout=0,
                                        decoder_pos_size=14, softmax=True)
        self.fc1 = nn.Linear(in_channels * 8, in_channels * 4)
        self.fc2 = nn.Linear(in_channels * 4, 1)
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None
    def forward(self, x):
        # Question here
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # cl_out1_a = self.avgpool1(x3)
        # cl_out1_m = self.maxpoll1(x3)

        x4 = self.down3(x3)
        x5 = self.down4(x4)

        cl_out_2_cl, x5 = self.task2(x5, x5)
        cl_out2_a = self.avgpool2(cl_out_2_cl)
        # cl_out2_m = self.maxpoll2(cl_out_2_cl)


        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        # cl_out = torch.flatten(torch.cat([cl_out1_a, cl_out1_m, cl_out2_a, cl_out2_m], dim=1), 1)
        cl_out = torch.flatten(cl_out2_a, 1)
        cl_out = self.fc1(cl_out)
        cl_out = self.fc2(cl_out)
        logits = self.outc(x)
        return logits, cl_out
class UNetTaskAligWeight(nn.Module):
    def __init__(self, n_channels=3, n_classes=9):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Question here
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpBlockAlig(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlockAlig(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlockAlig(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpBlockAlig(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1))

        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        # self.maxpoll1 = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        # self.maxpoll2 = nn.AdaptiveMaxPool2d((1, 1))
        self.task2 = TransformerDecoder(dim=in_channels * 8, depth=1, heads=8, dim_head=64,
                                        mlp_dim=2048, dropout=0,
                                        decoder_pos_size=14, softmax=True)
        self.fc1 = nn.Linear(in_channels * 8, in_channels * 4)
        self.fc2 = nn.Linear(in_channels * 4, 1)




    def forward(self, x):
        # Question here
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # cl_out1_a = self.avgpool1(x3)
        # cl_out1_m = self.maxpoll1(x3)

        x4 = self.down3(x3)
        out0 = self.down4(x4)

        cl_out_2_cl, out0 = self.task2(out0, out0)
        # cl_out2_a = self.avgpool2(cl_out_2_cl)
        # cl_out2_m = self.maxpoll2(cl_out_2_cl)

        out1 = self.up4(out0, x4)

        out2 = self.up3(out1, x3)

        out3 = self.up2(out2, x2)

        out4 = self.up1(out3, x1)

        # cl_out = torch.flatten(torch.cat([cl_out1_a, cl_out1_m, cl_out2_a, cl_out2_m], dim=1), 1)
        # cl_out = torch.flatten(cl_out2_a,1)
        # cl_out = self.fc1(cl_out)
        # cl_out = self.fc2(cl_out)
        # out = self.dialation(out4)
        logits = self.outc(out4)
        return logits
class deepBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(deepBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, (2, 2), 2)
        # if in_channels == 64:
        self.nConvs_1 = _make_nConv(in_channels+out_channels, 1, 1, 'Sigmoid')
        # else:
        #     self.nConvs_1 = _make_nConv(in_channels, out_channels, 1, 'Sigmoid')

        self.nConvs_2 = _make_nConv(out_channels, out_channels, 1, activation)

    def custom_function(self, x, device):
        a = torch.tensor(10, device=device, dtype=torch.float32, requires_grad=False)
        return torch.exp(-a * (x - 0.5) ** 2)
    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)
        x = self.nConvs_1(x)
        x = (1 - torch.abs(x - 0.5))
        x = skip_x * x
        return self.nConvs_2(x)
        # return self.nConvs(skip_x)
class Dilation(nn.Module):
    def __init__(self,in_channels):
        super(Dilation, self).__init__()
        # self.out1 = deepBlock(512,256,1)
        # self.out2 = deepBlock(256, 128, 1)
        # self.out3 = deepBlock(128, 64, 1)
        self.out4 = deepBlock(64, 64, 1)
    def forward(self, out0,out1,out2,out3,out4):
        # up1 = self.out1(out0,out1)
        # up2 = self.out2(up1, out2)
        # up3 = self.out3(up2, out3)
        up4 = self.out4(out3, out4)
        return up4

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels):
        super(DepthwiseSeparableConv, self).__init__()
        # self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
        #                            stride=stride, padding=padding, groups=in_channels)
        # self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.loss_function = AutomaticWeightedLoss(num=2)
        self.out = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))
        self.nConvs = _make_nConv(1, in_channels, 2, 'Relu')
    def sobel_edge_detection(self,image_tensor):
        with torch.no_grad():
            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image_tensor.device)
            sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image_tensor.device)
            edge_x = torch.nn.functional.conv2d(image_tensor, sobel_x, padding=1)
            edge_y = torch.nn.functional.conv2d(image_tensor, sobel_y, padding=1)
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
            return edge
    def custom_function(self, x, device):
        a = torch.tensor(20, device=device, dtype=torch.float32, requires_grad=False)
        return torch.exp(-a * (x - 0.5) ** 2)
    def forward(self, x):
        # v_1 = torch.tensor(1.0).to(x.device)
        # v_0 = torch.tensor(0.0).to(x.device)
        diff = self.out(x)
        diff = F.sigmoid(diff)
        # diff = torch.where( diff == 0.5,v_1 , v_0)
        diff = (1 - torch.abs(diff - 0.5))
        # diff = self.sobel_edge_detection(diff)
        # diff = torch.where(diff > 0.5, v_1, v_0)
        # diff = self.custom_function(diff,x.device)
        diff = self.nConvs(diff)
        # diff = torch.where(diff > 0.5, v_1, v_0)
        x = torch.cat([x,diff],dim=1)
        # x = self.loss_function(x,diff)

        return x

if __name__ == '__main__':
    # 创建一个示例模型
    model = UNetTaskAligWeight()
    print(model)
    # 如果有 GPU 可用，则将模型移动到 GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # # 打印模型的详细信息
    # # 输入形状 (3, 224, 224) 对应于彩色图像 (C, H, W)
    # summary(model, (3, 224, 224))
