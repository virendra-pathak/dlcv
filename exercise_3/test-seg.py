import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

#from dl4cv.classifiers.segmentation_nn import SegmentationNN
from dl4cv.classifiers.fcn16s import FCN16VGG as SegmentationNN
from dl4cv.data_utils import SegmentationData, label_img_to_rgb
from dl4cv.data_utils import OverfitSampler

#torch.set_default_tensor_type('torch.FloatTensor')

#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2


train_data = SegmentationData(image_paths_file='datasets/segmentation_data/train.txt')
val_data = SegmentationData(image_paths_file='datasets/segmentation_data/val.txt')

print("Train size: %i" % len(train_data))
print("Validation size: %i" % len(val_data))
print("Img size: ", train_data[0][0].size())
print("Segmentation size: ", train_data[0][1].size())

num_example_imgs = 3
plt.figure(figsize=(10, 5 * num_example_imgs))
for i, (img, target) in enumerate(train_data[:num_example_imgs]):
    # img
    plt.subplot(num_example_imgs, 2, i * 2 + 1)
    plt.imshow(img.numpy().transpose(1, 2, 0))
    plt.axis('off')
    if i == 0:
        plt.title("Input image")

    # target
    plt.subplot(num_example_imgs, 2, i * 2 + 2)
    plt.imshow(label_img_to_rgb(target.numpy()))
    plt.axis('off')
    if i == 0:
        plt.title("Target image")
plt.show()

#from dl4cv.classifiers.segmentation_nn import SegmentationNN
from dl4cv.classifiers.fcn16s import FCN16VGG as SegmentationNN
from dl4cv.solver import Solver
import torch.nn.functional as F


########################################################################
#                             YOUR CODE                                #
########################################################################
train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=1, sampler=OverfitSampler(20))
val_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=1, sampler=OverfitSampler(20))
model = SegmentationNN()
solver = Solver(optim_args={"lr": 1e-3})
solver.train(model, train_loader, val_loader, log_nth=1, num_epochs=1)