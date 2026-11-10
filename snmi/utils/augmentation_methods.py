from itertools import combinations
import numpy as np
from scipy import ndimage

from .process_methods import Resize

class RandomAugmentation:

    def __init__(self, img_suffix, lab_suffix, aug_rate=0.5):
        self.img_suffix = img_suffix
        self.lab_suffix = lab_suffix
        self.aug_rate = aug_rate

    def __call__(self, data_dict):
        r = np.random.rand()
        if r > self.aug_rate:
            return data_dict
        
        img = data_dict[self.img_suffix]
        lab = data_dict[self.lab_suffix]

        methods = [Rotate(np.random.randint(360), [0, 1]), flipup, fliplr]
        comb = []
        for r in range(1,len(methods)+1):
            comb += list(combinations(methods, r))
        selected = np.random.choice(comb, 1)[0]

        for m in selected:
            img = m(img, is_mask=False)
            lab = m(lab, is_mask=True)
        
        data_dict.update({self.img_suffix: img, 
                          self.lab_suffix: lab})
        
        return data_dict


class Rotate:
    def __init__(self, angle, ax):
        self.angle = angle
        self.ax = ax
    def __call__(self, img, is_mask):
        order = 0 if is_mask else 3
        img = ndimage.rotate(img, self.angle, self.ax, reshape=False, prefilter=True, order=order)
        return img

def flipup(img, is_mask):
    return np.flip(img, 0).copy()

def fliplr(img, is_mask):
    return np.flip(img, 1).copy()

class RandomCrop():
    def __init__(self, img_suffix, lab_suffix, tar_size, seg_suffix=None, resize=None):
        self.img_suffix = img_suffix
        self.lab_suffix = lab_suffix
        self.tar_size = tar_size
        self.seg_suffix = seg_suffix
        self.resize = resize

    def __call__(self, data_dict):
        imgo = data_dict[self.img_suffix].copy()
        mean_ind = np.mean(imgo)

        xs = np.random.randint(imgo.shape[0] - self.tar_size[0] + 1) 
        ys = np.random.randint(imgo.shape[1] - self.tar_size[1] + 1)
        img = imgo[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]] 
        while np.mean(img) < mean_ind / 10:
            xs = np.random.randint(imgo.shape[0] - self.tar_size[0] + 1) 
            ys = np.random.randint(imgo.shape[1] - self.tar_size[1] + 1)
            img = imgo[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]] 

        if self.resize is not None and self.tar_size != self.resize:
            img = Resize(self.resize)(img)
        data_dict.update({self.img_suffix: img})

        if self.seg_suffix is not None:
            seg = data_dict[self.seg_suffix]
            seg = seg[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]]   
            data_dict.update({self.seg_suffix: seg}) 
        
        return data_dict  

class RandomForegroundCrop():
    def __init__(self, img_suffix, seg_suffix, tar_size, foreground_ratio):
        self.img_suffix = img_suffix
        self.seg_suffix = seg_suffix
        self.tar_size = tar_size
        self.foreground_ratio = foreground_ratio

    def __call__(self, data_dict):
        imgo = data_dict[self.img_suffix].copy()
        labo = data_dict[self.seg_suffix].copy()
        count_fore = np.sum(labo==255)

        xs = np.random.randint(imgo.shape[0] - self.tar_size[0] + 1) 
        ys = np.random.randint(imgo.shape[1] - self.tar_size[1] + 1)
        img = imgo[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]] 
        lab = labo[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]]
        while np.sum(lab==255) < count_fore * self.foreground_ratio:
            xs = np.random.randint(imgo.shape[0] - self.tar_size[0] + 1) 
            ys = np.random.randint(imgo.shape[1] - self.tar_size[1] + 1)
            img = imgo[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]] 
            lab = labo[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]]

        data_dict.update({self.img_suffix: img})

        return data_dict  


class IntensityJitter:
    """Photometric jitter that survives per-patch min_max normalization.

    Brightness/contrast scaling is pointless when the pre-processing applies min_max
    afterwards (it re-stretches the patch and undoes any affine change), so this only
    applies a gamma curve and additive noise, which min_max cannot undo.
    """

    def __init__(self, gamma_range=(0.7, 1.4), noise_std=0.02, rate=0.5):
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.rate = rate

    def __call__(self, img):
        if np.random.rand() > self.rate:
            return img
        x = np.asarray(img, np.float32)
        lo, hi = float(x.min()), float(x.max())
        scale = hi - lo if hi > lo else 1.0
        x01 = (x - lo) / scale
        if self.gamma_range is not None:
            g = np.random.uniform(*self.gamma_range)
            x01 = np.power(np.clip(x01, 0, 1), g)
        if self.noise_std:
            x01 = x01 + np.random.normal(0, self.noise_std, x01.shape).astype(np.float32)
        return np.clip(x01, 0, 1) * scale + lo


class RandomDihedral2D:
    """Random element of the 8-fold dihedral group, applied to several keys at once.

    Why not `RandomAugmentation`: that class rotates by an arbitrary angle on axes
    [0, 1], which (a) interpolates and leaves zero-filled corners that never occur in
    real tissue, (b) assumes every array is [H, W] and so would rotate the channel axis
    of a one-hot [2, H, W] label, and (c) leaves the weight map untouched, pairing an
    augmented image with an unaugmented confidence map.

    This class works on the last two axes, so [H, W], [C, H, W] and [C, H, W, ...] all
    behave, applies the identical geometric transform to every listed key, and uses
    only rot90/flip, which are exact (no interpolation, no padding).

    Parameters
    ----------
    image_keys: keys that get the geometric transform and the optional jitter.
    map_keys: keys that get the geometric transform only (labels, weight maps).
    aug_rate: probability of applying a geometric transform at all.
    jitter: an IntensityJitter (or any callable on an array), applied to image_keys.
    """

    def __init__(self, image_keys, map_keys=(), aug_rate=1.0, jitter=None):
        self.image_keys = list(image_keys)
        self.map_keys = list(map_keys)
        self.aug_rate = aug_rate
        self.jitter = jitter

    def __call__(self, data_dict):
        k = np.random.randint(4)
        flip = np.random.rand() < 0.5
        apply_geom = np.random.rand() <= self.aug_rate

        for key in self.image_keys + self.map_keys:
            if key not in data_dict:
                continue
            x = data_dict[key]
            if apply_geom:
                if k:
                    x = np.rot90(x, k, axes=(-2, -1))
                if flip:
                    x = np.flip(x, -1)
                x = np.ascontiguousarray(x)   # torch cannot take negative strides
            if self.jitter is not None and key in self.image_keys:
                x = self.jitter(x)
            data_dict[key] = x
        return data_dict
