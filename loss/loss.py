import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Loss, self).__init__()

    def forward(self, logits, targets,y5,y4,y3,y2):
        x2 = F.upsample(targets, scale_factor=0.5)
        x3 = F.upsample(x2, scale_factor=0.5)
        x4 = F.upsample(x3, scale_factor=0.5)
        x5 = F.upsample(x4, scale_factor=0.5)

        def calculate_loss(logits, targets):
            p = logits.view(-1, 1)
            t = targets.view(-1, 1)

            # Binary Cross-Entropy Loss
            loss1 = F.binary_cross_entropy_with_logits(p, t, reduction='mean')
            smooth = 1

            # 计算 Dice 系数损失
            probs = torch.sigmoid(logits)
            m1 = probs.view(targets.size(0), -1)
            m2 = targets.view(targets.size(0), -1)
            intersection = (m1 * m2)
            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            loss2 = 1 - score.sum() / targets.size(0)
            
            return loss1 + loss2

        loss_y5 = calculate_loss(y5, x5)
        loss_y4 = calculate_loss(y4, x4)
        loss_y3 = calculate_loss(y3, x3)
        loss_y2 = calculate_loss(y2, x2)
        loss_y1 = calculate_loss(logits, targets)

        total_loss = loss_y5 + loss_y4 + loss_y3 + loss_y2 + loss_y1

        return total_loss
