import torch.nn as nn
import torch.nn.functional as F
class DiceLoss(nn.Module):
    def __init__(self,eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self,logits,targets):
        probs = F.softmax(logits,dim=1)
        targets_onehot = F.one_hot(targets,num_classes=probs.shape[1]).permute(0,3,1,2).float()
        num = 2 * (probs * targets_onehot).sum(dim=(2,3))

        den = (probs + targets_onehot).sum(dim=(2,3))
        dice = (num +self.eps)/(den + self.eps)
        return 1 - dice.mean()

class AttentionGate(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super().__inin__()
        self.W_g = nn.Conv2d(F_g,F_int,kernel_size=1,bias=True)
        self.W_x = nn.Conv2d(F_l,F_int,kernel_size=1,bias=True)
        self.psi = nn.Conv2d(F_int,1,kernel_size=1,bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x,g):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g,size=x.shape[2:],mode='bilinear',align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        alpha = self.sigmoid(psi)
        return x * alpha