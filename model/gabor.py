import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
from torchvision import transforms
from torch.nn import functional as F
class Gabor:
    a00 = 0.3
    a02 = 0.078
    a10 = 0.23
    a12 = 0.078
    a20 = 0.24342268924547819
    a21 = 0.20476744424496821

    finalimg = None
        
    M = torch.Tensor([[a00, 1 - a00 - a02, a02],
        [a10, 1 - a10 - a12, a12],
        [a20, a21, 1 - a20 - a21]]).to("cuda:0")

    #Minv = [[2542.56835937500000, -2326.53173828125000, 44.17734527587891],
    #        [-698.04852294921875, 1012.71777343750000, -76.22974395751953],
    #        [-827.36480712890625, 709.83502197265625, 451.38488769531250]]

    Minv = torch.Tensor([[11.031566901960783, -9.866943921568629, -0.16462299647058826],
            [-3.254147380392157,  4.418770392156863,  -0.16462299647058826],
            [-3.6588512862745097, 2.7129230470588235, 1.9459282392156863]]).to("cuda:0")
    def __init__(self,img):
        self.finalimg = self.inverse_gabor(img)
    def toXYB(self,img):
        # rgb = np.asarray(img)
        rgb = torch.Tensor(img).to("cuda:0")
        # xyb = np.zeros((rgb.shape[0], rgb.shape[1], 3))
        xyb = torch.zeros((rgb.shape[0], rgb.shape[1], 3)).to("cuda:0")
        
        bias = 0.0037930732552754493
        # eps = np.array([bias, bias, bias]).reshape(-1, 1)
        eps = torch.Tensor([bias, bias, bias]).reshape(-1, 1).to("cuda:0")
        xybfilt = torch.Tensor([[0.5,-0.5,0],[0.5,0.5,0],[0,0,1]]).to("cuda:0")
        gamma = None
        xyblist = []
        innnerxyblist = []
        innnerxyb = None
        for r in range(rgb.shape[0]):
            for c in range(rgb.shape[1]):
                pixel = rgb[r, c].reshape(-1, 1)

                # mix = np.matmul(self.M, pixel) + eps
                mix = torch.matmul(self.M, pixel) + eps
                # gamma = np.cbrt(mix) + eps
                gamma = torch.pow(mix,1/3) + eps
                # xyb[r, c, 0] = (gamma[0] - gamma[1]) / 2
                # xyb[r, c, 1] = (gamma[0] + gamma[1]) / 2
                # xyb[r, c, 2] = gamma[2]
                a = torch.matmul(xybfilt,gamma)#.reshape(1,3)
                innnerxyblist.append(torch.squeeze(a,dim=1))
            innnerxyb = torch.stack(innnerxyblist)
            xyblist.append(innnerxyb)
            innnerxyblist=[]
        xyb = torch.stack(xyblist)
        return xyb
    def toRGB(self,img):
        # bias = np.array([-0.00100549, -0.00100549, -0.00094081]).reshape(-1, 1)
        bias = torch.Tensor([-0.00100549, -0.00100549, -0.00094081]).reshape(-1, 1).to("cuda:0")
        # rgb = np.zeros((img.shape[0], img.shape[1], 3))
        rgb = torch.zeros((img.shape[0], img.shape[1], 3)).to("cuda:0")
        
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                xyb = img[r, c]
                r_gamma = xyb[1] + xyb[0]
                g_gamma = xyb[1] - xyb[0]
                b_gamma = xyb[2]
                
                # mix = np.array([r_gamma ** 3, g_gamma ** 3, b_gamma ** 3]).reshape(-1, 1) + bias
                mix = torch.Tensor([r_gamma ** 3, g_gamma ** 3, b_gamma ** 3]).reshape(-1, 1).to("cuda:0") + bias
                # pixel = np.matmul(self.Minv, mix).reshape(1, -1)[0]
                pixel = torch.matmul(self.Minv, mix).reshape(1, -1)[0]
                
                rgb[r, c, 0] = pixel[0]
                rgb[r, c, 1] = pixel[1]
                rgb[r, c, 2] = pixel[2]
                
        return rgb
    def inverse_gabor(self,img):
        #rgb = np.asarray(img)
        norm = transforms.Normalize([-1,-1,-1],[2,2,2])
        img = norm(img)
        img_xyb = self.toXYB(img)
        # print(img_xyb)
        
        c = 1
        L = 0.00083458437774987476
        R = 0.016176494530216929
        D = 0.004512465323949319
        r = -0.092359145662814029
        d = -0.039253623634014627

        weights = [[D, L, R, L, D],
                [L, d, r, d, L],
                [R, r, c, r, R],
                [L, d, r, d, L],
                [D, L, R, L, D]]

        # kernel = np.array(weights)
        # kernel *= -1 / (1 - 2 * (4 * c + 4 * r + 4 * d + 4 * R + 4 * D + 8 * L))
        kernel1 = torch.Tensor(weights).to("cuda:0")
        kernel = torch.mul((-1 / (1 - 2 * (4 * c + 4 * r + 4 * d + 4 * R + 4 * D + 8 * L))),kernel1)
        
        x = signal.convolve2d(img_xyb[:, :, 0].detach().cpu().numpy(), kernel.cpu().numpy()[::-1, ::-1], boundary='symm', mode='same')
        y = signal.convolve2d(img_xyb[:, :, 1].detach().cpu().numpy(), kernel.cpu().numpy()[::-1, ::-1], boundary='symm', mode='same')
        b = signal.convolve2d(img_xyb[:, :, 2].detach().cpu().numpy(), kernel.cpu().numpy()[::-1, ::-1], boundary='symm', mode='same')
        
        # result = np.zeros((img_xyb.shape[0], img_xyb.shape[1], 3))
        result = torch.ones((img_xyb.shape[0], img_xyb.shape[1], 3)).to("cuda:0")
        xyblist = []
        xyblist.append(torch.Tensor(x).to("cuda:0"))
        xyblist.append(torch.Tensor(y).to("cuda:0"))
        xyblist.append(torch.Tensor(b).to("cuda:0"))
        xybtensor = torch.stack(xyblist).reshape(128,128,3)
        # for r in range(img_xyb.shape[0]):
        #     for c in range(img_xyb.shape[1]):
        #         result[r, c, 0] = x[r, c]
        #         result[r, c, 1] = y[r, c]
        #         result[r, c, 2] = b[r, c]
        result = torch.mul(result,xybtensor)
        # restored = abs(self.toRGB(result))
        restored = torch.abs(self.toRGB(result))
        
        # for i in range(3):
        #     restored[:, :, i] = restored[:, :, i] / (restored[:, :, i].max() / 255.0)
        rmax = 1/(torch.max(restored[:, :, 0]).item()/255.0)
        gmax = 1/(torch.max(restored[:, :, 1]).item()/255.0)
        bmax = 1/(torch.max(restored[:, :, 2]).item()/255.0)
        rten=torch.Tensor(np.ones((restored.shape[0],restored.shape[0]))*rmax).to("cuda:0")
        gten=torch.Tensor(np.ones((restored.shape[0],restored.shape[0]))*gmax).to("cuda:0")
        bten=torch.Tensor(np.ones((restored.shape[0],restored.shape[0]))*bmax).to("cuda:0")
        converlist = []
        converlist.append(rten)
        converlist.append(gten)
        converlist.append(bten)
        converten = torch.stack(converlist).reshape(128,128,3)
        restored = torch.mul(converten,restored)
        norm1 = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        restored = norm1(restored)
        out_img = Image.fromarray(np.uint8(restored.detach().cpu())).convert('RGB')
        return out_img
# #raw = Image.open('000000000025.jpg')
# #raw = Image.open('000000000036.jpg')
# raw = Image.open('000000000073.jpg')
# #raw = Image.open('huo_blured.png')
# #raw = Image.open('4x4r1.jpg')
# trans = transforms.ToTensor()
# toPIL = transforms.ToPILImage()

# img = trans(np.array(raw))
# out_raw = toPIL(img)
# plt.imshow(out_raw)
# plt.show()
# sharpened = inverse_gabor(raw)
# sharpened
