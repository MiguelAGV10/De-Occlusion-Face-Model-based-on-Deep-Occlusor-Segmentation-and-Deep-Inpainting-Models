
import torch
import torch.nn as nn

from common_loss import VGG19

### Class for calculating the content (perptual), style, and recontrusction losses. There is no mask here as it is self-supervised learning. 
### The model by itself learns what are the regions to be inpainted.
class total_loss(nn.Module):
    """
    Class for calculating the loss during training and validation without explicit guidance of the regions to be inpainted
    Losses:
        L1. Using torch L1 loss
        Content (perceptual). Using VGG feautures
        Style. Using VGG features' gram matrices
    """
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0]):
        super(total_loss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights
        self.criterion_local = torch.nn.L1Loss()
        
    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        style_loss = 0.0
        prefix_style = [2,3,4,5]
        posfix_style = [2,4,4,2]
        prefix = [1, 2, 3, 4]
        for i in range(4):
            content_loss = content_loss + self.weights[i] * self.criterion(
                x_vgg[f'relu{prefix[i]}_1'], y_vgg[f'relu{prefix[i]}_1'])
        
        for pre, pos in list(zip(prefix_style,posfix_style)):
             style_loss = style_loss + self.criterion(self.compute_gram(x_vgg[f'relu{pre}_{pos}']), self.compute_gram(y_vgg[f'relu{pre}_{pos}']))
        
        
        rec_loss_local = self.criterion_local(x,y)

        return content_loss, style_loss, rec_loss_local
