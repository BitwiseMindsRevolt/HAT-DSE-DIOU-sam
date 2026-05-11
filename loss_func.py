import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, focal=False, weight=None, reduce=True):
        super(MultiCrossEntropyLoss, self).__init__()
        self.focal = focal
        self.weight= weight
        self.reduce = reduce

    def forward(self, input, target):
        #IN: input: unregularized logits [B, C] target: multi-hot representaiton [B, C]
        target_sum = torch.sum(target, dim=1)
        target_div = torch.where(target_sum != 0, target_sum, torch.ones_like(target_sum)).unsqueeze(1)
        target = target/target_div
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        if not self.focal:
            if self.weight is None:
                output = torch.sum(-target * logsoftmax(input), 1)
            else:
                output = torch.sum(-target * logsoftmax(input) /self.weight, 1)
        else:
            softmax = nn.Softmax(dim=1).to(input.device)
            p = softmax(input)
            output = torch.sum(-target * (1 - p)**2 * logsoftmax(input), 1)
            
        if self.reduce:
            return torch.mean(output)
        else:
            return output
    

def cls_loss_func(y,output, use_focal=False, weight=None, reduce=True):
    input_size=y.size()
    y = y.float().cuda()
    if weight is not None:
        weight = weight.cuda()
    loss_func = MultiCrossEntropyLoss(focal=use_focal, weight=weight, reduce=reduce)
    
    y=y.reshape(-1,y.size(-1))
    output=output.reshape(-1,output.size(-1))
    loss = loss_func(output,y)
    
    if not reduce:
        loss = loss.reshape(input_size[:-1])
    
    return loss


def regress_loss_func(y,output):
    y = y.float().cuda()
    
    #y=y.unsqueeze(-1)
    y=y.reshape(-1,y.size(-1))
    output=output.reshape(-1,output.size(-1))
    
    bgmask= y[:,1] < -1e2
    
    fg_logits = output[~bgmask]
    bg_logits = output[bgmask]
    
    fg_target = y[~bgmask]
    bg_target = y[bgmask]
    
    loss = nn.functional.l1_loss(fg_logits,fg_target)
    #loss = nn.functional.smooth_l1_loss(fg_logits, fg_target, beta=0.5)
    
        
    if(loss.isnan()):
        return torch.tensor([0.0], requires_grad=True).cuda()
    return loss


def diou_loss_func(y, output, anchors):
    # y, output: [B, A, 2] with channel 0 = end-offset / anchor_len,
    # channel 1 = log(gt_len / anchor_len). Background rows have y[...,1] = -1e3.
    y = y.float().cuda()
    output = output.float()

    A = y.shape[1]
    anc = torch.tensor(anchors, dtype=output.dtype, device=output.device).view(1, A)

    fg = y[..., 1] > -1e2
    if fg.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=output.device)

    # Decode to (st, ed) in frame units. The current-timestamp offset and the anchor
    # end cancel inside IoU and center distance, so we keep them at 0.
    pred_ed = anc * output[..., 0]
    pred_len = anc * torch.exp(torch.clamp(output[..., 1], max=10.0))
    pred_st = pred_ed - pred_len

    tgt_ed = anc * y[..., 0]
    tgt_len = anc * torch.exp(y[..., 1])
    tgt_st = tgt_ed - tgt_len

    inter_st = torch.maximum(pred_st, tgt_st)
    inter_ed = torch.minimum(pred_ed, tgt_ed)
    inter = (inter_ed - inter_st).clamp(min=0)
    union = (pred_len + tgt_len - inter).clamp(min=1e-6)
    iou = inter / union

    pred_c = (pred_st + pred_ed) * 0.5
    tgt_c = (tgt_st + tgt_ed) * 0.5
    enc_st = torch.minimum(pred_st, tgt_st)
    enc_ed = torch.maximum(pred_ed, tgt_ed)
    enc_d = (enc_ed - enc_st).clamp(min=1e-6)

    diou = iou - (pred_c - tgt_c) ** 2 / (enc_d ** 2)
    loss = (1.0 - diou)[fg]

    if loss.numel() == 0 or torch.isnan(loss).any():
        return torch.tensor(0.0, requires_grad=True, device=output.device)
    return loss.mean()


def suppress_loss_func(y,output):
    y = y.float().cuda()

    #y=y.unsqueeze(-1)
    y=y.reshape(-1,y.size(-1))
    output=output.reshape(-1,output.size(-1))

    loss = nn.functional.binary_cross_entropy(output,y)

    return loss
