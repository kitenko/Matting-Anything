import os
import math
import numbers
import random
import logging
import numpy as np
import albumentations as A

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision import transforms

from utils import CONFIG

import warnings
warnings.filterwarnings("ignore")
import cv2

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test", real_world_aug=False):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.phase = phase
        if real_world_aug:
            self.RWA = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.4, p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.ImageCompression(quality_range=(60, 95), p=0.3)
            ])
        else:
            self.RWA = None

    def get_box_from_alpha(self, alpha_final):
        bi_mask = np.zeros_like(alpha_final)
        bi_mask[alpha_final > 0.5] = 1
        fg_set = np.where(bi_mask != 0)
        if len(fg_set[1]) == 0 or len(fg_set[0]) == 0:
            x_min = random.randint(1, 511)
            x_max = random.randint(1, 511) + x_min
            y_min = random.randint(1, 511)
            y_max = random.randint(1, 511) + y_min
        else:
            x_min = np.min(fg_set[1])
            x_max = np.max(fg_set[1])
            y_min = np.min(fg_set[0])
            y_max = np.max(fg_set[0])
        bbox = np.array([x_min, y_min, x_max, y_max])
        return bbox

    def __call__(self, sample):
        # Конвертация изображения из GBR в RGB
        image, alpha, trimap = sample['image'][:,:,::-1], sample['alpha'], sample['trimap']
        
        alpha = np.clip(alpha, 0, 1)

        bbox = self.get_box_from_alpha(alpha)

        if self.phase == 'train' and self.RWA is not None and np.random.rand() < 0.5:
            image = np.clip(image, 0, 255).astype(np.uint8)
            augmented = self.RWA(image=image)
            image = augmented['image']

        # Меняем порядок осей, чтобы получить формат C x H x W
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 2
        trimap[trimap >= 85] = 1
        # Нормализуем изображение
        image /= 255.

        if self.phase == "train":
            fg = sample['fg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg'] = torch.from_numpy(fg).sub_(self.mean).div_(self.std)
            bg = sample['bg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['bg'] = torch.from_numpy(bg).sub_(self.mean).div_(self.std)
            del sample['image_name']
        
        sample['boxes'] = torch.from_numpy(bbox).to(torch.float)[None,...]

        sample['image'], sample['alpha'], sample['trimap'] = \
            torch.from_numpy(image), torch.from_numpy(alpha), torch.from_numpy(trimap).to(torch.long)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)

        if CONFIG.model.trimap_channel == 3:
            sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2,0,1).float()
        elif CONFIG.model.trimap_channel == 1:
            sample['trimap'] = sample['trimap'][None,...].float()
        else:
            raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")

        return sample


class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int32) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha

        return sample


    @staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix


class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha == 0):
            return sample
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat > 1] = 2 - sat[sat > 1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        val_jitter = np.random.rand()*(1.1 - val_bar)/5 - (1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val > 1] = 2 - val[val > 1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg*255

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if np.random.uniform(0, 1) < self.prob:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)
        sample['fg'], sample['alpha'] = fg, alpha

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=( CONFIG.data.crop_size, CONFIG.data.crop_size)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        fg, alpha, trimap, name = sample['fg'],  sample['alpha'], sample['trimap'], sample['image_name']
        bg = sample['bg']
        h, w = trimap.shape
        bg = cv2.resize(bg, (w, h), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0]/h if h < w else 1.1*self.output_size[1]/w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:
                fg = cv2.resize(fg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (int(w*ratio), int(h*ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap = cv2.resize(trimap, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                bg = cv2.resize(bg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
                h, w = trimap.shape
        small_trimap = cv2.resize(trimap, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4,
                                                       self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)

        fg_crop = fg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        alpha_crop = alpha[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        bg_crop = bg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        trimap_crop = trimap[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]

        if len(np.where(trimap==128)[0]) == 0:
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                                "left_top: {}".format(name, left_top))
            fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        
        sample.update({'fg': fg_crop, 'alpha': alpha_crop, 'trimap': trimap_crop, 'bg': bg_crop})
        return sample


class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,100)]

    def __call__(self, sample):
        alpha = sample['alpha']
        h, w = alpha.shape

        max_kernel_size = max(30, int((min(h,w) / 2048) * 30))

        ### generate trimap
        fg_mask = (alpha + 1e-5).astype(np.int32).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int32).astype(np.uint8)
        # Ensure kernel size is within available range
        kernel_size = np.random.randint(1, min(max_kernel_size, len(self.erosion_kernels)-1) + 1)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[kernel_size])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[kernel_size])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w,h), interpolation=cv2.INTER_NEAREST)
        sample['trimap'] = trimap

        return sample


class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        bg[bg < 0 ] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image
        return sample


class Composite_Seg(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        image = fg
        sample['image'] = image
        return sample


class DataGenerator(Dataset):
    def __init__(self, phase="train"):
        self.phase = phase
        self.crop_size = CONFIG.data.crop_size
        self.pha_ratio = CONFIG.data.pha_ratio

        # Initialize lists of file paths
        self.coco_bg = self._load_paths(CONFIG.data.coco_bg, "coco_bg")
        self.coco_num = len(self.coco_bg)
        self.bg20k_bg = self._load_paths(CONFIG.data.bg20k_bg, "bg20k_bg")
        self.bg20k_num = len(self.bg20k_bg)

        self.d646_fg = self._load_paths(CONFIG.data.d646_fg, "d646_fg")
        self.d646_pha = self._load_paths(CONFIG.data.d646_pha, "d646_pha")

        self.aim_fg = self._load_paths(CONFIG.data.aim_fg, "aim_fg")
        self.aim_pha = self._load_paths(CONFIG.data.aim_pha, "aim_pha")

        self.am2k_fg = self._load_paths(CONFIG.data.am2k_fg, "am2k_fg")
        self.am2k_pha = self._load_paths(CONFIG.data.am2k_pha, "am2k_pha")
        
        self.rim_img = [os.path.join(CONFIG.data.rim_img, name) for name in sorted(os.listdir(CONFIG.data.rim_img))]
        self.rim_pha = [os.path.join(CONFIG.data.rim_pha, name) for name in sorted(os.listdir(CONFIG.data.rim_pha))]

        # Проверка соответствия длины списков и корректировка масок
        self._validate_and_adjust_paths()
        
        self.d646_num = len(self.d646_fg)
        self.aim_num = len(self.aim_fg)
        self.am2k_num = len(self.am2k_fg)
        self.rim_num = len(self.rim_img)

        # Определение трансформаций
        self.transform_imagematte = transforms.Compose([
            RandomAffine(degrees=30, scale=[0.8, 1.5], shear=10, flip=0.5),
            GenTrimap(),
            RandomCrop((self.crop_size, self.crop_size)),
            RandomJitter(),
            Composite(),
            ToTensor(phase="train", real_world_aug=CONFIG.data.real_world_aug)
        ])

        self.transform_spd = transforms.Compose([
            RandomAffine(degrees=30, scale=[0.8, 1.5], shear=10, flip=0.5),
            GenTrimap(),
            RandomCrop((self.crop_size, self.crop_size)),
            Composite_Seg(),
            ToTensor(phase="train", real_world_aug=CONFIG.data.real_world_aug)
        ])

    def _load_paths(self, directory, name):
        if not os.path.isdir(directory):
            logging.error(f"Directory for {name} does not exist: {directory}")
            return []
        files = sorted(os.listdir(directory))
        full_paths = [os.path.join(directory, file) for file in files if self._is_image(file)]
        return full_paths

    def _is_image(self, filename):
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        return filename.lower().endswith(IMG_EXTENSIONS)

    def _validate_and_adjust_paths(self):
        """
        Обеспечивает соответствие между списками fg изображений и pha масок на основе имен файлов.
        Удаляет маски, для которых нет соответствующих fg изображений.
        """
        datasets = [
            ('am2k_fg', 'am2k_pha'),
            ('d646_fg', 'd646_pha'),
            ('aim_fg', 'aim_pha'),
            # ('human2k_fg', 'human2k_pha'),
            ('rim_img', 'rim_pha')
        ]

        for fg_name, pha_name in datasets:
            fg_list = getattr(self, fg_name, [])
            pha_list = getattr(self, pha_name, [])

            # Создаём словарь для быстрого поиска масок по базовому имени файла
            pha_dict = {os.path.splitext(os.path.basename(path))[0]: path for path in pha_list}

            matched_fg = []
            matched_pha = []

            for fg_path in fg_list:
                base_name = os.path.splitext(os.path.basename(fg_path))[0]
                pha_path = pha_dict.get(base_name)
                if pha_path:
                    matched_fg.append(fg_path)
                    matched_pha.append(pha_path)
                else:
                    logging.warning(f"Не найдена маска для fg изображения: {fg_path}")

            # Обновляем списки fg и pha с учётом сопоставленных путей
            setattr(self, fg_name, matched_fg)
            setattr(self, pha_name, matched_pha)

            # Логируем количество лишних масок, которые не имеют соответствующих fg изображений
            extra_masks = set(os.path.splitext(os.path.basename(p))[0] for p in pha_list) - set(os.path.splitext(os.path.basename(p))[0] for p in matched_fg)
            if extra_masks:
                logging.warning(f"{len(extra_masks)} масок не имеют соответствующих fg изображений для {fg_name}.")

            # Дополнительно проверяем, что после сопоставления количество масок равно количеству fg изображений
            if len(matched_pha) != len(matched_fg):
                logging.error(f"Количество сопоставленных масок ({len(matched_pha)}) не совпадает с количеством fg изображений ({len(matched_fg)}) для {fg_name}.")
                raise ValueError(f"Несоответствие количества масок и fg изображений для {fg_name} и {pha_name}.")

    def __getitem__(self, idx):
        if random.random() < 0.5:
            bg = cv2.imread(self.coco_bg[idx])
        else:
            bg = cv2.imread(self.bg20k_bg[idx % self.bg20k_num])
        
        if random.random() < 0.5:
            # if random.random() < 0.25:
                # fg = cv2.imread(self.human2k_fg[idx % self.human2k_num])
                # alpha = cv2.imread(self.human2k_pha[idx % self.human2k_num], 0).astype(np.float32)/255

                # fg, alpha = self._composite_fg_human2k(fg, alpha, idx)
                # image_name = os.path.split(self.human2k_fg[idx % self.human2k_num])[-1]

            if random.random() < 0.5:
                fg = cv2.imread(self.am2k_fg[idx % self.am2k_num])
                alpha = cv2.imread(self.am2k_pha[idx % self.am2k_num], 0).astype(np.float32)/255

                fg, alpha = self._composite_fg_am2k(fg, alpha, idx)
                image_name = os.path.split(self.am2k_fg[idx % self.am2k_num])[-1]
            
            elif random.random() < 0.75:
                fg = cv2.imread(self.d646_fg[idx % self.d646_num])
                alpha = cv2.imread(self.d646_pha[idx % self.d646_num], 0).astype(np.float32)/255

                fg, alpha = self._composite_fg_646(fg, alpha, idx)
                image_name = os.path.split(self.d646_fg[idx % self.d646_num])[-1]

            else:
                fg = cv2.imread(self.aim_fg[idx % self.aim_num])
                alpha = cv2.imread(self.aim_pha[idx % self.aim_num], 0).astype(np.float32)/255

                fg, alpha = self._composite_fg_aim(fg, alpha, idx)
                image_name = os.path.split(self.aim_fg[idx % self.aim_num])[-1]
            
            sample = {'fg': fg, 'alpha': alpha, 'bg': bg, 'image_name': image_name}
            sample = self.transform_imagematte(sample)
        else:
            fg = cv2.imread(self.rim_img[idx % self.rim_num])
            alpha = cv2.imread(self.rim_pha[idx % self.rim_num], 0).astype(np.float32)/255

            if np.random.rand() < 0.25:
                fg = cv2.resize(fg, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

            image_name = os.path.split(self.rim_img[idx % self.rim_num])[-1]
            sample = {'fg': fg, 'alpha': alpha, 'bg': bg, 'image_name': image_name}
            sample = self.transform_spd(sample)

        return sample

    def __len__(self):
        # Определение длины датасета как количества fg изображений
        return self.coco_num  # Или другое, в зависимости от основного набора данных

    def safe_read(self, path, flags=cv2.IMREAD_COLOR):
        """
        Безопасное чтение изображения. Если изображение не может быть загружено, выбрасывает исключение.
        """
        img = cv2.imread(path, flags)
        if img is None:
            logging.error(f"Не удалось загрузить изображение: {path}")
            raise FileNotFoundError(f"Не удалось загрузить изображение: {path}")
        return img

    def _composite_fg_am2k(self, fg, alpha, idx):
        if np.random.rand() < 0.5:
            try:
                idx2 = np.random.randint(self.am2k_num) + idx
                fg2_path = self.am2k_fg[idx2 % self.am2k_num]
                pha2_path = self.am2k_pha[idx2 % self.am2k_num]
                fg2 = self.safe_read(fg2_path, cv2.IMREAD_COLOR)
                alpha2 = self.safe_read(pha2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
                h, w = alpha.shape
                fg2 = cv2.resize(fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha2 = cv2.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

                alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
                if np.any(alpha_tmp < 1):
                    fg = fg.astype(np.float32) * alpha[:, :, None] + fg2.astype(np.float32) * (1 - alpha[:, :, None])
                    alpha = alpha_tmp
                    fg = fg.astype(np.uint8)
            except Exception as e:
                logging.error(f"Ошибка в _composite_fg_am2k: {e}")

        if np.random.rand() < 0.25:
            fg = cv2.resize(fg, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(alpha, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg, alpha

    def _composite_fg_646(self, fg, alpha, idx):
        if np.random.rand() < 0.5:
            try:
                idx2 = np.random.randint(self.d646_num) + idx
                fg2_path = self.d646_fg[idx2 % self.d646_num]
                pha2_path = self.d646_pha[idx2 % self.d646_num]
                fg2 = self.safe_read(fg2_path, cv2.IMREAD_COLOR)
                alpha2 = self.safe_read(pha2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
                h, w = alpha.shape
                fg2 = cv2.resize(fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha2 = cv2.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

                alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
                if np.any(alpha_tmp < 1):
                    fg = fg.astype(np.float32) * alpha[:, :, None] + fg2.astype(np.float32) * (1 - alpha[:, :, None])
                    alpha = alpha_tmp
                    fg = fg.astype(np.uint8)
            except Exception as e:
                logging.error(f"Ошибка в _composite_fg_646: {e}")

        if np.random.rand() < 0.25:
            fg = cv2.resize(fg, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(alpha, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg, alpha

    def _composite_fg_aim(self, fg, alpha, idx):
        if np.random.rand() < 0.5:
            try:
                idx2 = np.random.randint(self.aim_num) + idx
                fg2_path = self.aim_fg[idx2 % self.aim_num]
                pha2_path = self.aim_pha[idx2 % self.aim_num]
                fg2 = self.safe_read(fg2_path, cv2.IMREAD_COLOR)
                alpha2 = self.safe_read(pha2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
                h, w = alpha.shape
                fg2 = cv2.resize(fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha2 = cv2.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

                alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
                if np.any(alpha_tmp < 1):
                    fg = fg.astype(np.float32) * alpha[:, :, None] + fg2.astype(np.float32) * (1 - alpha[:, :, None])
                    alpha = alpha_tmp
                    fg = fg.astype(np.uint8)
            except Exception as e:
                logging.error(f"Ошибка в _composite_fg_aim: {e}")

        if np.random.rand() < 0.25:
            fg = cv2.resize(fg, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(alpha, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg, alpha
