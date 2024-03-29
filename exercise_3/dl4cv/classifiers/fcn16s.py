import torch
from torch import nn
from torchvision import models
import numpy as np

#from ..utils import get_upsampling_weight
#from .config import vgg16_caffe_path

# Credit: https://github.com/ZijunDeng/pytorch-semantic-segmentation
class FCN16VGG(nn.Module):
    def __init__(self, num_classes=23, pretrained=True):
        super(FCN16VGG, self).__init__()
        vgg = models.vgg16()
        print("Virendra why parameters is NULL ?")
        print(vgg.parameters())
        if pretrained:
            #/media/virendra/data/study/3sem/DeepLearning/Tutorial/exercise1/dl4cv/exercise_3/models/vgg16-397923af.pth
            #vgg.load_state_dict(torch.load(vgg16_caffe_path))
            vgg.load_state_dict(torch.load("/media/virendra/data/study/3sem/DeepLearning/Tutorial/exercise1/dl4cv/exercise_3/models/vgg16-397923af.pth"))
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = False
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = False

        self.features4 = nn.Sequential(*features[: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, bias=False)
        self.upscore2.weight.data.copy_(self.get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore16.weight.data.copy_(self.get_upsampling_weight(num_classes, num_classes, 32))
        self.softmax = torch.nn.Softmax2d()

    def forward(self, x):
        x_size = x.size()
        pool4 = self.features4(x)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * pool4)
        upscore16 = self.upscore16(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                   + upscore2)
        #return upscore16[:, :, 27: (27 + x_size[2]), 27: (27 + x_size[3])].contiguous()

        upscore16 = upscore16[:, :, 27: (27 + x_size[2]), 27: (27 + x_size[3])]
        #upscore16 = self.softmax(upscore16)
        return upscore16

    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
        weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
        return torch.from_numpy(weight).float()
    
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

