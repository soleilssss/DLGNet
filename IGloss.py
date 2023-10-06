import math
import torch
import torch.nn as nn


class IGLoss(nn.Module):
    def __init__(self, feat_dim, num_classes,  alpha=0.1, lambda_=0.01):
        super(IGLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.lambda_ = lambda_
        self.means = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.variance = nn.Parameter(torch.randn(num_classes))
        nn.init.xavier_uniform_(self.means, gain=math.sqrt(2.0))
        nn.init.uniform_(self.variance, a=0.9, b=1.1)

    def forward(self, feat, labels=None):
        batch_size= feat.size()[0]

        XY = torch.matmul(feat, torch.transpose(self.means, 0, 1))  #全连接batchsize*numclasses
        XX = torch.sum(feat ** 2, dim=1, keepdim=True)  #一个值
        YY = torch.sum(torch.transpose(self.means, 0, 1)**2, dim=0, keepdim=True)  #[1*numclasses]
        var =torch.diag(self.variance ** 2)
        neg_sqr = -0.5 * (XX - 2.0 * XY + YY)
        neg_sqr_dist = torch.mm(neg_sqr, var)

        if labels is None:
            psudo_labels = torch.argmax(neg_sqr_dist, dim=1)
            means_batch = torch.index_select(self.means, dim=0, index=psudo_labels)
            likelihood_reg_loss = self.lambda_ * (torch.sum((feat - means_batch)**2) / 2) * (1. / batch_size)
            return neg_sqr_dist, likelihood_reg_loss, self.means

        labels_reshped = labels.view(labels.size()[0], -1)

        if torch.cuda.is_available():
            ALPHA = torch.zeros(batch_size, self.num_classes).cuda().scatter_(1, labels_reshped, self.alpha)
            K = ALPHA + torch.ones([batch_size, self.num_classes]).cuda()
        else:
            ALPHA = torch.zeros(batch_size, self.num_classes).scatter_(1, labels_reshped, self.alpha)
            K = ALPHA + torch.ones([batch_size, self.num_classes])

        logits_with_margin = torch.mul(neg_sqr_dist, K)
        means_batch = torch.index_select(self.means, dim=0, index=labels)
        likelihood_reg_loss = self.lambda_ * (torch.sum((feat - means_batch)**2) / 2) * (1. / batch_size)
        return logits_with_margin, likelihood_reg_loss, self.means