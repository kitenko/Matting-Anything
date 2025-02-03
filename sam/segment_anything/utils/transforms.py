# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    
    
    
class ResizeMaskWithAspect:
    """
    Класс для исправления масштабирования маски.
    
    Предполагается, что на вход подаётся маска размера 256×256 (или другого квадратного размера),
    которая была получена после ресайза с некорректным соотношением сторон.
    Метод apply_mask возвращает маску, изменённую таким образом, чтобы её размер соответствовал
    исходным пропорциям изображения (передаются height_origin и width_origin),
    при этом максимальная сторона будет равна target_size (например, 256).
    """

    def __init__(self, target_size: int = 256) -> None:
        """
        Аргументы:
            target_size (int): максимальный размер (длина) наибольшей стороны после ресайза.
        """
        self.target_size = target_size

    @staticmethod
    def get_new_size(height_origin: int, width_origin: int, target_size: int) -> Tuple[int, int]:
        """
        Вычисляет новый размер (new_h, new_w) для изображения с исходным размером (height_origin, width_origin),
        чтобы максимальная сторона стала равна target_size и сохранилось исходное соотношение сторон.
        
        Аргументы:
            height_origin (int): исходная высота изображения.
            width_origin (int): исходная ширина изображения.
            target_size (int): требуемая длина наибольшей стороны.
            
        Возвращает:
            Tuple[int, int]: новый размер (new_h, new_w).
        """
        scale = target_size / max(height_origin, width_origin)
        new_h = int(round(height_origin * scale))
        new_w = int(round(width_origin * scale))
        return new_h, new_w

    def apply_mask(
        self,
        mask: torch.Tensor,
        height_origin: int,
        width_origin: int,
        mode: str = "nearest",
        **kwargs,
    ) -> torch.Tensor:
        """
        Ресайзит маску с квадратного размера (например, 256×256) к новому размеру,
        вычисленному на основе исходных размеров изображения (height_origin, width_origin)
        с сохранением пропорций. При этом максимальная сторона будет равна self.target_size.
        
        Аргументы:
            mask (torch.Tensor): Маска, ожидается тензор размера (B, C, H, W) или (C, H, W).
                                   Обычно H=W=256.
            height_origin (int): Исходная высота изображения.
            width_origin (int): Исходная ширина изображения.
            mode (str): Режим интерполяции. Для масок рекомендуется "nearest".
            **kwargs: Дополнительные аргументы для F.interpolate.
        
        Возвращает:
            torch.Tensor: Маска с новым размером, соответствующим исходному соотношению сторон.
        """
        # Вычисляем новый размер
        new_h, new_w = self.get_new_size(height_origin, width_origin, self.target_size)

        # Если маска имеет форму (C, H, W), добавляем батч-измерение
        need_unsqueeze = False
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
            need_unsqueeze = True

        # Применяем интерполяцию
        resized_mask = F.interpolate(mask, size=(new_h, new_w), mode=mode, **kwargs)

        # Если исходно маска была без батч-измерения, убираем его обратно
        if need_unsqueeze:
            resized_mask = resized_mask.squeeze(0)
        return resized_mask