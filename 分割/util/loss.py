from monai.losses import DiceLoss




import torch.nn as nn
import torch.nn.functional as F
import torch


from .lovasz import _lovasz_hinge, _lovasz_softmax
from typing import Optional, Union

class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # b h w -> b 1 h w
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    def forward(self, net_output, target, weight=None, sigmoid=False):
        if sigmoid:
            net_output = torch.sigmoid(net_output) # b 1 h w
        assert net_output.size() == target.size(), 'predict {} & target {} shape do not match'.format(net_output.size(),
                                                                                                      target.size())
        dice_loss = self._dice_loss(net_output[:, 0], target[:, 0])
        return dice_loss
class Mask_DC_and_BCE_loss(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.6):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_DC_and_BCE_loss, self).__init__()
        self.ce =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight
    def forward(self, net_output, target):
        low_res_logits = net_output
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3], low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target)
        loss_dice = self.dc(low_res_logits, target, sigmoid=True)

        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss

class DC_and_BCE_loss(nn.Module):
    def __init__(self, classes=2, dice_weight=0.6):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()

        self.ce = nn.BCEWithLogitsLoss()
        self.dc = DiceLoss(sigmoid=True)
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        # low_res_logits = net_output['low_res_logits']
        # if len(target.shape) == 4:
        #     target = target[:, 0, :, :]
        loss_ce = self.ce(net_output, target)
        loss_dice = self.dc(net_output, target)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.6, 0.4], gamma=1.5, reduction='mean',):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        # 将索引张量移到与被索引张量相同的设备上
        self.alpha = self.alpha.to(target.device)
        alpha = self.alpha[target]   # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=1.5, alpha=0.2, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn
class TverskyLoss(nn.Module):
    def __init__(self, alpha,beta,apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return -tversky
class FocalTversky_loss(nn.Module):
    """
    paper: https://arxiv.org/pdf/1810.07842.pdf
    author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py#L65
    数据集的类别分布：观察数据集中各个类别的样本数是否平衡，如果存在类别不平衡，可以尝试使用较小的gamma来解决这个问题。
    难易样本的分布：了解数据集中是否存在难易样本，如果存在较多难易样本，可以尝试使用较大的gamma来加权处理这些样本。
    交叉验证：使用交叉验证或其他评估方法来评估不同gamma值下模型的性能，并选择表现较好的gamma值。
    """

    def __init__(self, gamma=2,alpha = 0.6 ,beta = 0.7):
        super(FocalTversky_loss, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(alpha,beta)

    def forward(self, net_output, target):
        tversky_loss = 1 + self.tversky(net_output, target)  # = 1-tversky(net_output, target)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky




# binary loss
class BinaryJaccardLoss(nn.Module):
    """
    binary Jaccard loss,iou loss
    """

    def __init__(self):
        super(BinaryJaccardLoss, self).__init__()
        self.smooth = 1e-5
        self.eps = 1e-7

    def forward(self, y_pred_logits, y_true):
        # y_pred = F.logsigmoid(y_pred_logits).exp()
        y_pred = torch.sigmoid(y_pred_logits)
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        y_pred = y_pred.float().contiguous().view(bs, num_classes, -1)
        y_true = y_true.float().contiguous().view(bs, num_classes, -1)
        intersection = (y_pred * y_true).sum()
        dsc = (intersection + self.smooth) / (y_pred.sum() + y_true.sum() - intersection + self.smooth).clamp_min(
            self.eps)
        loss = 1. - dsc
        return loss.mean()


class BinaryDiceLoss(nn.Module):
    """
    binary dice loss
    """

    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1e-5
        self.eps = 1e-7

    def forward(self, y_pred_logits, y_true):
        # y_pred = F.logsigmoid(y_pred_logits).exp()
        y_pred = torch.sigmoid(y_pred_logits)
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        y_pred = y_pred.float().contiguous().view(bs, num_classes, -1)
        y_true = y_true.float().contiguous().view(bs, num_classes, -1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth).clamp_min(self.eps)
        loss = 1. - dsc
        return loss.mean()


class BinaryELDiceLoss(nn.Module):
    """
    binary Exponential Logarithmic Dice loss
    """

    def __init__(self):
        super(BinaryELDiceLoss, self).__init__()
        self.smooth = 1e-5
        self.eps = 1e-7

    def forward(self, y_pred_logits, y_true):
        y_pred = torch.sigmoid(y_pred_logits)
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        y_pred = y_pred.float().contiguous().view(bs, num_classes, -1)
        y_true = y_true.float().contiguous().view(bs, num_classes, -1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth).clamp_min(self.eps)
        return torch.clamp((torch.pow(-torch.log(dsc + self.smooth), 0.3)).mean(), 0, 2)


class BinarySSLoss(nn.Module):
    """
    binary Sensitivity-Specifity loss
    """

    def __init__(self):
        super(BinarySSLoss, self).__init__()
        self.smooth = 1e-5
        self.r = 0.1  # weight parameter in SS paper

    def forward(self, y_pred_logits, y_true):
        # y_pred = F.logsigmoid(y_pred_logits).exp()
        y_pred = torch.sigmoid(y_pred_logits)
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        y_pred = y_pred.float().contiguous().view(bs, num_classes, -1)
        y_true = y_true.float().contiguous().view(bs, num_classes, -1)
        bg_y_true = 1 - y_true
        squared_error = (y_pred - y_true) ** 2
        specificity_part = (squared_error * y_true).sum() / (self.smooth + y_true.sum())
        sensitivity_part = (squared_error * bg_y_true).sum() / (self.smooth + bg_y_true.sum())
        ss = self.r * specificity_part + (1 - self.r) * sensitivity_part
        return ss.mean()


class BinaryTverskyLoss(nn.Module):
    """
    binary tversky loss,paper: https://arxiv.org/pdf/1706.05721.pdf
    """

    def __init__(self):
        super(BinaryTverskyLoss, self).__init__()
        self.smooth = 1e-5
        self.alpha = 0.3
        self.beta = 0.7

    def forward(self, y_pred, y_true):
        # y_pred = F.logsigmoid(y_pred_logits).exp()
        # y_pred = torch.sigmoid(y_pred_logits)
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        y_pred = y_pred.float().contiguous().view(bs, num_classes, -1)
        y_true = y_true.float().contiguous().view(bs, num_classes, -1)
        bg_true = 1 - y_true
        bg_pred = 1 - y_pred
        tp = (y_pred * y_true).sum()
        fp = (y_pred * bg_true).sum()
        fn = (bg_pred * y_true).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.clamp((1 - tversky).mean(), 0, 2)


class BinaryCrossEntropyLoss(nn.Module):
    """
    HybridLoss
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
    pytorch推荐使用binary_cross_entropy_with_logits,
    将sigmoid层和binaray_cross_entropy合在一起计算比分开依次计算有更好的数值稳定性，这主要是运用了log-sum-exp技巧。
    """

    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, y_pred_logits, y_true):
        bs = y_true.size(0)
        num_classes = y_pred_logits.size(1)
        y_pred_logits = y_pred_logits.float().contiguous().view(bs, num_classes, -1)
        y_true = y_true.float().contiguous().view(bs, num_classes, -1)
        bce = F.binary_cross_entropy_with_logits(y_pred_logits.float(), y_true.float())
        return bce


# class BinaryFocalLoss(nn.Module):
#     """
#     binary focal loss
#     """
#
#     def __init__(self, alpha=0.25, gamma=2):
#         super(BinaryFocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, y_pred_logits, y_true):
#         """
#         https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/4
#         """
#         bs = y_true.size(0)
#         num_classes = y_pred_logits.size(1)
#         y_pred_logits = y_pred_logits.float().contiguous().view(bs, num_classes, -1)
#         y_true = y_true.float().contiguous().view(bs, num_classes, -1)
#         BCE_loss = F.binary_cross_entropy_with_logits(y_pred_logits.float(), y_true.float(), reduction='none')
#         pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         """
#         Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#         """
#         # p = torch.sigmoid(y_pred_logits)
#         # p_t = p * y_true + (1 - p) * (1 - y_true)
#         # loss = BCE_loss * ((1 - p_t) ** self.gamma)
#         # if self.alpha >= 0:
#         #     alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
#         #     focal_loss = alpha_t * loss
#         return focal_loss.mean()
class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):


        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.to(labels.view(-1).device)
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum.cuda()
class BinaryCrossEntropyDiceLoss(nn.Module):
    """
    binary Dice loss + BCE loss
    """

    def __init__(self):
        super(BinaryCrossEntropyDiceLoss, self).__init__()

    def forward(self, y_pred_logits, y_true):
        diceloss = BinaryDiceLoss()
        dice = diceloss(y_pred_logits, y_true)
        bceloss = BinaryCrossEntropyLoss()
        bce = bceloss(y_pred_logits, y_true)
        return bce*0.5 + dice*0.5


class MCC_Loss(nn.Module):
    """
    Compute Matthews Correlation Coefficient Loss for image segmentation task. It only supports binary mode.
    Calculates the proposed Matthews Correlation Coefficient-based loss.
    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        return 1 - mcc


class BinaryLovaszLoss(nn.Module):
    def __init__(self, per_image: bool = False, ignore_index: Optional[Union[int, float]] = None):
        super(BinaryLovaszLoss).__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_hinge(logits, target, per_image=self.per_image, ignore_index=self.ignore_index)


# mutil loss

class MutilCrossEntropyLoss(nn.Module):
    def __init__(self, alpha):
        super(MutilCrossEntropyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred_logits, y_true):
        Batchsize, Channel = y_pred_logits.shape[0], y_pred_logits.shape[1]
        y_pred_logits = y_pred_logits.float().contiguous().view(Batchsize, Channel, -1)
        y_true = y_true.long().contiguous().view(Batchsize, -1)
        y_true_onehot = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
        y_true_onehot = y_true_onehot.permute(0, 2, 1).float()  # H, C, H*W
        mask = y_true_onehot.sum((0, 2)) > 0
        loss = F.cross_entropy(y_pred_logits.float(), y_true.long(), weight=mask.to(y_pred_logits.dtype))
        return loss


class MutilFocalLoss(nn.Module):
    """
    """

    def __init__(self, alpha, gamma=2, torch=True):
        super(MutilFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.torch = torch

    def forward(self, y_pred_logits, y_true):
        if torch:
            Batchsize, Channel = y_pred_logits.shape[0], y_pred_logits.shape[1]
            y_pred_logits = y_pred_logits.float().contiguous().view(Batchsize, Channel, -1)
            y_true = y_true.long().contiguous().view(Batchsize, -1)
            y_true_onehot = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
            y_true_onehot = y_true_onehot.permute(0, 2, 1).float()  # H, C, H*W
            mask = y_true_onehot.sum((0, 2)) > 0
            CE_loss = nn.CrossEntropyLoss(reduction='none', weight=mask.to(y_pred_logits.dtype))
            logpt = CE_loss(y_pred_logits.float(), y_true.long())
            pt = torch.exp(-logpt)
            loss = (((1 - pt) ** self.gamma) * logpt).mean()
            return loss


class MutilDiceLoss(nn.Module):
    """
        multi label dice loss with weighted
        Y_pred: [None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_pred is softmax result
        Y_gt:[None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_gt is one hot result
        alpha: tensor shape (C,) where C is the number of classes,eg:[0.1,1,1,1,1,1]
        :return:
        """

    def __init__(self, alpha):
        super(MutilDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred_logits, y_true):
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1
        # y_pred = y_pred_logits.log_softmax(dim=1).exp()
        y_pred = torch.softmax(y_pred_logits, dim=1)
        Batchsize, Channel = y_pred.shape[0], y_pred.shape[1]
        y_pred = y_pred.float().contiguous().view(Batchsize, Channel, -1)
        y_true = y_true.long().contiguous().view(Batchsize, -1)
        y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1)  # H, C, H*W
        smooth = 1.e-5
        eps = 1e-7
        assert y_pred.size() == y_true.size()
        intersection = torch.sum(y_true * y_pred, dim=(0, 2))
        denominator = torch.sum(y_true + y_pred, dim=(0, 2))
        gen_dice_coef = ((2. * intersection + smooth) / (denominator + smooth)).clamp_min(eps)
        loss = - gen_dice_coef
        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        mask = y_true.sum((0, 2)) > 0
        loss *= mask.to(loss.dtype)
        return (loss * self.alpha).sum() / torch.count_nonzero(mask)


class MutilCrossEntropyDiceLoss(nn.Module):
    """
    Mutil Dice loss + CE loss
    """

    def __init__(self, alpha):
        super(MutilCrossEntropyDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred_logits, y_true):
        diceloss = MutilDiceLoss(self.alpha)
        dice = diceloss(y_pred_logits, y_true)
        celoss = MutilCrossEntropyLoss(self.alpha)
        ce = celoss(y_pred_logits, y_true)
        return ce + dice


class MutilELDiceLoss(nn.Module):
    """
        multi label Exponential Logarithmic Dice loss with weighted
        Y_pred: [None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_pred is softmax result
        Y_gt:[None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_gt is one hot result
        alpha: tensor shape (C,) where C is the number of classes,eg:[0.1,1,1,1,1,1]
        :return:
        """

    def __init__(self, alpha):
        super(MutilELDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred_logits, y_true):
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1
        # y_pred = y_pred_logits.log_softmax(dim=1).exp()
        y_pred = torch.softmax(y_pred_logits, dim=1)
        Batchsize, Channel = y_pred.shape[0], y_pred.shape[1]
        y_pred = y_pred.float().contiguous().view(Batchsize, Channel, -1)
        y_true = y_true.long().contiguous().view(Batchsize, -1)
        y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1)  # H, C, H*W
        smooth = 1.e-5
        eps = 1e-7
        assert y_pred.size() == y_true.size()
        intersection = torch.sum(y_true * y_pred, dim=(0, 2))
        denominator = torch.sum(y_true + y_pred, dim=(0, 2))
        gen_dice_coef = ((2. * intersection + smooth) / (denominator + smooth)).clamp_min(eps)
        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        mask = y_true.sum((0, 2)) > 0
        gen_dice_coef *= mask.to(gen_dice_coef.dtype)
        dice = gen_dice_coef * self.alpha
        return torch.clamp((torch.pow(-torch.log(dice + smooth), 0.3)).sum() / torch.count_nonzero(mask), 0, 2)


class MutilSSLoss(nn.Module):
    """
        multi label SS loss with weighted
        Y_pred: [None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_pred is softmax result
        Y_gt:[None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_gt is one hot result
        alpha: tensor shape (C,) where C is the number of classes,eg:[0.1,1,1,1,1,1]
        :return:
        """

    def __init__(self, alpha):
        super(MutilSSLoss, self).__init__()
        self.alpha = alpha
        self.smooth = 1.e-5

    def forward(self, y_pred_logits, y_true):
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1
        # y_pred = y_pred_logits.log_softmax(dim=1).exp()
        y_pred = torch.softmax(y_pred_logits, dim=1)
        Batchsize, Channel = y_pred.shape[0], y_pred.shape[1]
        y_pred = y_pred.float().contiguous().view(Batchsize, Channel, -1)
        y_true = y_true.long().contiguous().view(Batchsize, -1)
        y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1)  # H, C, H*W
        assert y_pred.size() == y_true.size()
        bg_true = 1 - y_true
        squared_error = (y_true - y_pred) ** 2
        specificity = torch.sum(squared_error * y_true, dim=(0, 2)) / (torch.sum(y_true, dim=(0, 2)) + self.smooth)
        sensitivity = torch.sum(squared_error * bg_true, dim=(0, 2)) / (torch.sum(y_true, dim=(0, 2)) + self.smooth)
        ss = self.r * specificity + (1 - self.r) * sensitivity
        mask = y_true.sum((0, 2)) > 0
        ss *= mask.to(ss.dtype)
        return (ss * self.alpha).sum() / torch.count_nonzero(mask)


class MutilTverskyLoss(nn.Module):
    """
        multi label TverskyLoss loss with weighted
        Y_pred: [None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_pred is softmax result
        Y_gt:[None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_gt is one hot result
        alpha: tensor shape (C,) where C is the number of classes,eg:[0.1,1,1,1,1,1]
        :return:
        """

    def __init__(self, alpha):
        super(MutilTverskyLoss, self).__init__()
        self.alpha = alpha
        self.smooth = 1.e-5

    def forward(self, y_pred_logits, y_true):
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1
        # y_pred = y_pred_logits.log_softmax(dim=1).exp()
        y_pred = torch.softmax(y_pred_logits, dim=1)
        Batchsize, Channel = y_pred.shape[0], y_pred.shape[1]
        y_pred = y_pred.float().contiguous().view(Batchsize, Channel, -1)
        y_true = y_true.long().contiguous().view(Batchsize, -1)
        y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1)  # H, C, H*W
        assert y_pred.size() == y_true.size()
        bg_true = 1 - y_true
        bg_pred = 1 - y_pred
        tp = torch.sum(y_pred * y_true, dim=(0, 2))
        fp = torch.sum(y_pred * bg_true, dim=(0, 2))
        fn = torch.sum(bg_pred * y_true, dim=(0, 2))
        tversky = -(tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        # tversky loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        mask = y_true.sum((0, 2)) > 0
        tversky *= mask.to(tversky.dtype)
        return (tversky * self.alpha).sum() / torch.count_nonzero(mask)


class LovaszLoss(nn.Module):
    """
    mutil LovaszLoss
    """

    def __init__(self, per_image=False, ignore=None):
        super(LovaszLoss, self).__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_softmax(logits, target, per_image=self.per_image, ignore_index=self.ignore)
