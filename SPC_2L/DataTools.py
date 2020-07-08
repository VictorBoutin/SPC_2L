import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
from torchvision.datasets.folder import default_loader, find_classes, make_dataset, IMG_EXTENSIONS
from math import exp
import math
import matplotlib.pyplot as plt
#from MLCSC.Monitor import DisplayDico

class DataBase(object):
    def __init__(self, name, path, batch_size, shuffle=False, reshaped_size=None, img_size=None, gray_scale=False,
                 do_mask=False, mask_params=None,
                 do_LCN=True, LCN_params=None,
                 do_whitening=False, whitening_params = None,
                 do_z_score=False,
                 num_workers=0,
                 path_target=None,
                 normalize=False,
                 return_idx=False):

        self.param = {'name': name,
                      'path': path,
                      'batch_size': batch_size}
        self.name = name
        self.path = path
        self.batch_size=batch_size
        self.path_target = path_target
        list_of_transforms = []
        if name == 'from_ImageFolder':
            if gray_scale :
                list_of_transforms.append(transforms.Grayscale())

            if reshaped_size is not None:
                list_of_transforms.append(transforms.Resize(size=reshaped_size))
                img_size=reshaped_size
          
            list_of_transforms.append(transforms.ToTensor())
            list_of_transforms.append(to_cuda())

            
            if do_LCN:
                if LCN_params is not None:
                    list_of_transforms.append(LCN(**LCN_params))
                else:
                    list_of_transforms.append(LCN())
            if do_whitening:
                if whitening_params is not None:
                    list_of_transforms.append(whitening(img_size, **whitening_params))
                else:
                    list_of_transforms.append(whitening(img_size))
            if do_z_score:
                list_of_transforms.append(z_score())
            if do_mask:
                if mask_params is not None:
                    list_of_transforms.append(mask(img_size, **mask_params))
                else:
                    list_of_transforms.append(mask(img_size))
        else:
            list_of_transforms.append(transforms.ToTensor())

        self.size = [None, int(batch_size), None, None, None]
        if name == 'from_tensor':
            data_tensor = torch.load(path)
            if self.path_target is not None :
                label_tensor = torch.load(self.path_target)
            else :
                label_tensor = torch.zeros(data_tensor.size()[0])

            data_set = torch.utils.data.TensorDataset(data_tensor, label_tensor)
        elif name == 'from_ImageFolder':
            transform = transforms.Compose(list_of_transforms)
            data_set = MyImageLoader(root=path, transform=transform, retun_idx=return_idx)
        elif name == 'CIFAR10_tr':
            if normalize == True:
                list_of_transforms.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
            transform = transforms.Compose(list_of_transforms)
            data_set = torchvision.datasets.CIFAR10(root=path, train=True,
                                       download=False,transform=transform)
        elif name == 'CIFAR10_te':
            if normalize == True:
                list_of_transforms.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
            transform = transforms.Compose(list_of_transforms)
            data_set = torchvision.datasets.CIFAR10(root=path, train=False,
                                       download=False,transform=transform)
        elif name == 'MNIST_tr':
            if normalize is True :
                list_of_transforms.append(transforms.Normalize((0.1307,), (0.3081,)))
            transform = transforms.Compose(list_of_transforms)
            data_set = torchvision.datasets.MNIST(root=path, train=True, download=False, transform=transform)
        elif name == 'MNIST_te':
            if normalize is True:
                list_of_transforms.append(transforms.Normalize((0.1307,), (0.3081,)))
            transform = transforms.Compose(list_of_transforms)
            data_set = torchvision.datasets.MNIST(root=path, train=False, download=False, transform=transform)
        else :
            raise NameError('{0} is not a builtin dataset'.format(name))
        self.size[-1] = data_set.__getitem__(0)[0].size()[-1]
        self.size[-2] = data_set.__getitem__(0)[0].size()[-2]
        self.size[-3] = data_set.__getitem__(0)[0].size()[-3]
        self.num_elements = data_set.__len__()

        if data_set.__len__() % self.batch_size != 0:
            print('warning: Not all batches have the same size, the last one will be dropped...')
            self.data = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
        else :
            self.data = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.size[0] = int(self.data.__len__())

    def ShowSample(self, nb_sample):
        first_batch = next(iter(self.data))[0]
        if nb_sample > first_batch.size()[0]:
            nb_sample = first_batch.size()[0]

       # DisplayDico(first_batch[0:nb_sample, :, :, :])


def normalize_tensor_(tensor):
    """
    Function that returns the l2-normalized tensor according to its first dimension.

    Args:
        tensor: a 4-dimensional torch tensor

    Returns:
        Normalized tensor

    TO DO:
        Trigger an error if the input tensor is not 4 dimensional
    """

    tensor_size = tensor.size()
    tensor = tensor.view(tensor_size[0], -1)
    norm = tensor.pow(2).sum(-1, keepdim=True).sqrt()
    tensor /= norm
    return tensor.view(tensor_size)

class to_cuda(object):

    '''
    transform input to CUDA tensor before pre-processing
    if a GPU is available
    '''

    def __call__(self, img):

        img = img.float()
        if torch.cuda.is_available():
            img=img.cuda()

        return img

'''
class LCN(object):

    ''''''

    Local Contrast Normalization as defined in Jarret et al. 2009 (http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf)

    ''''''

    def __init__(self, kernel_size=9, sigma=0.5, rgb=False):

        if kernel_size % 2 == 0:
            raise ('kernel size must be odd...')
        x = torch.from_numpy(np.linspace(-1, 1, kernel_size)).unsqueeze(1).expand(kernel_size, kernel_size).float()
        y = torch.from_numpy(np.linspace(-1, 1, kernel_size)).unsqueeze(0).expand(kernel_size, kernel_size).float()
        r_2 = x.pow(2) + y.pow(2)
        self.kernel_size = kernel_size
        self.gaussian_k = torch.exp(-r_2/sigma).unsqueeze(0).unsqueeze(0)
        if rgb:
            self.groups = 3
            self.gaussian_k = self.gaussian_k.expand(3, 1, kernel_size, kernel_size)
        else:
            self.groups = 1
            #self.gaussian_k = self.gaussian_k.expand(1,3,kernel_size,kernel_size)
        self.gaussian_k = self.gaussian_k / self.gaussian_k.sum()
        if torch.cuda.is_available():
            self.gaussian_k=self.gaussian_k.cuda()

    def __call__(self, img):
        # subtractive step
        img = img.unsqueeze(0)
        img_pad = F.pad(img, ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2))
        gaus_map = F.conv2d(img_pad, self.gaussian_k, groups=self.groups)
        img_sub = img - gaus_map
        # divisive step
        img_pad = F.pad(img_sub, ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2))
        img_sigma = F.conv2d(img_pad.pow(2), self.gaussian_k, groups=self.groups).sqrt()
        c = img_sigma.view(img_sigma.size()[0], -1).mean(-1, keepdim=True).unsqueeze(-1).unsqueeze(-1).expand_as(img_sigma)
        #img_sigma = (F.relu(img_sigma - c) > 0).float() * img_sigma + (1 - (F.relu(img_sigma - c) > 0).float()) * c
        img_sigma = F.relu(img_sigma - c) * img_sigma + (1 - F.relu(img_sigma - c)) * c
        img_div = img_sub / img_sigma

        return img_div.squeeze(0)
'''

class LCN(object):

    '''

    Local Contrast Normalization as defined in Jarret et al. 2009 (http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf)

    '''

    def __init__(self, kernel_size=9, sigma=0.5, rgb=False, param_shit=1):
        if kernel_size % 2 == 0:
            raise ('kernel size must be odd...')
        x = torch.from_numpy(np.linspace(-1, 1, kernel_size)).unsqueeze(1).expand(kernel_size, kernel_size).float()
        y = torch.from_numpy(np.linspace(-1, 1, kernel_size)).unsqueeze(0).expand(kernel_size, kernel_size).float()
        r_2 = x.pow(2) + y.pow(2)
        self.kernel_size = kernel_size
        self.gaussian_k = torch.exp(-r_2/sigma).unsqueeze(0).unsqueeze(0)
        if rgb:
            self.gaussian_k = self.gaussian_k.expand(3, 3, kernel_size, kernel_size)
        self.gaussian_k = self.gaussian_k / self.gaussian_k.sum()
        if torch.cuda.is_available():
            self.gaussian_k=self.gaussian_k.cuda()

    def __call__(self, img):
        # subtractive step
        img = img.unsqueeze(0)
        img_pad = F.pad(img, ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2))
        gaus_map = F.conv2d(img_pad, self.gaussian_k)
        img_sub = img - gaus_map
        # divisive step
        img_pad = F.pad(img_sub, ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2))
        img_sigma = F.conv2d(img_pad.pow(2), self.gaussian_k).sqrt()
        c = img_sigma.view(img_sigma.size()[0], -1).mean(-1, keepdim=True).unsqueeze(-1).unsqueeze(-1).expand_as(img_sigma)
        img_sigma = (F.relu(img_sigma - c) > 0).float() * img_sigma + (1 - (F.relu(img_sigma - c) > 0).float()) * c
        img_div = img_sub / img_sigma

        return img_div.squeeze(0)


class z_score(object):

    '''

    Image per Image z-score normalization

    Image = (Image-mean(Image))/std(Image)

    '''

    def __call__(self, img):

        img = img - img.mean()
        img = img / img.std()

        return img

class whitening(object):

    '''

    Whitening filter as in Olshausen&Field 1995

    R(f) = f * exp((f/f_0)^n)

    '''

    def __init__(self, img_size, f_0=0.5, n=4):
        self.f_0=f_0
        self.n = n
        if not torch.backends.mkl.is_available():
            raise Exception('MKL not found, sorry cannot use whitening filter')
        dim_x = img_size[0]
        dim_y = img_size[1]
        f_x = torch.from_numpy(np.linspace(-0.5, 0.5, dim_x)).unsqueeze(1).expand(dim_x, dim_y).float()
        f_y = torch.from_numpy(np.linspace(-0.5, 0.5, dim_y)).unsqueeze(0).expand(dim_x, dim_y).float()
        self.f = (f_x.pow(2) + f_y.pow(2)).sqrt().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        if torch.cuda.is_available():
            self.f = self.f.cuda()


    def __call__(self, img):

        img = img.unsqueeze(0)
        img_f = torch.rfft(img, 2, onesided=False)
        filt = self.f * torch.exp(-(self.f / self.f_0).pow(self.n))
        img_f_ = img_f * filt
        img = torch.irfft(img_f_, 2, onesided=False)

        return img.squeeze(0)

class mask(object):

    '''

    Masking Image borders to avoid artifacts

    '''

    def __init__(self, img_size, n=10):

        dim_x = img_size[0]
        dim_y = img_size[1]
        x = torch.from_numpy(np.linspace(-1, 1, dim_x)).unsqueeze(1).expand(dim_x, dim_y).float().unsqueeze(0)
        mask_x = 1 - x.abs().pow(n)
        y = torch.from_numpy(np.linspace(-1, 1, dim_y)).unsqueeze(0).expand(dim_x, dim_y).float().unsqueeze(0)
        mask_y = 1 - y.abs().pow(n)
        self.mask = mask_x * mask_y
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, img):

        return img*self.mask

class MyImageLoader(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, retun_idx=False):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

        try:
            imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        except:
            imgs = make_dataset(root, class_to_idx)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.retun_idx = retun_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.retun_idx:
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)


def norm(tensor):
    """
    Function that returns the l2 norm of a tensor according to its first dimension.

    Args:
        tensor: a 4-dimensional torch tensor

    Returns:
        l2 norm of the tensor

    TO DO:
        Trigger an error if the input tensor is not 4 dimensional
    """
    tensor = tensor.view(tensor.size()[0], -1)
    ##norm = tensor.pow(2).sum(-1, keepdim=True).sqrt() ## WARNING : These functions gives only approximation
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
    return norm.unsqueeze(-1).unsqueeze(-1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    #window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def pnsr(img1, img2):
    diff = F.mse_loss(img1, img2)
    return 10 * torch.log10(1 / diff)

def gaussian_kernel(size, sigma=2):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(size[-1])
    x_grid = x_coord.repeat(size[-1]).view(size[-1], size[-2])
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (size[-1] - 1)/2.
    variance = sigma**2.

    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel/= gaussian_kernel.max()
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, size[-2], size[-1])
    gaussian_kernel = gaussian_kernel.repeat(size[0], size[1], 1, 1)

    return gaussian_kernel.cuda()

def to_img(x):
    val_max = x.abs().max(-1, keepdim=True)[0].max(-2, keepdim=True)[0].expand_as(x)
    x += val_max
    x /= 2 * val_max
    x.clamp(0, 1)
    return x

def show(img):
    #img = img.view(img.size()[0],img.size()[2],img.size()[3])
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.axis('off')
    plt.show()