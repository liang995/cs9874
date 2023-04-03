import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from model.gabor import Gabor
from torchvision import transforms
import torch 
import time
from model.gabormod import Model

class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(ConvBNRelu(3, config.discriminator_channels))
        for _ in range(config.discriminator_blocks-1):
            layers.append(ConvBNRelu(config.discriminator_channels, config.discriminator_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(config.discriminator_channels, 1)

    def forward(self, image):
        # an attmept of utilizing inverse gabor but did not work well to converge at all
        # sharpened = None
        # combined = []
        # if image.shape[1]==3:
        #     for i in range(image.shape[0]):
        #         T1= torch.narrow(image, 0,0 ,1).to("cuda:0")
        #         if T1.shape[0] == 1:
        #             toTensor = transforms.ToTensor()
        #             sqT1 = torch.squeeze(T1,dim=0)
        #             transT1 = torch.reshape(sqT1,(128,128,3))
        #             # if transT1.requires_grad == True:
        #             #     sharpened = Gabor(transT1.detach().cpu().numpy())
        #             # else:
        #             #     sharpened = Gabor(transT1.cpu())
        #             sharpened = Gabor(transT1)
        #             combined.append(toTensor(sharpened.finalimg).reshape(3,128,128).to("cuda:0"))
        #     combined = torch.stack(combined)
        #     X = self.before_linear(combined)
        # else:
        #     X = self.before_linear(image)
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X