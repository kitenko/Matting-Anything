import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

from utils import CONFIG
from networks import m2ms, ops
import sys
sys.path.insert(0, './segment-anything')
from sam2.torch_sam_image_predictor import SAM2TorchPredictor

from segment_anything.utils.transforms import ResizeMaskWithAspect

class sam2_m2m(nn.Module):
    def __init__(self, seg, m2m):
        super().__init__()
        if m2m not in m2ms.__all__:
            raise NotImplementedError("Unknown M2M {}".format(m2m))
        self.m2m = m2ms.__dict__[m2m](nc=256)
        self.predictor = SAM2TorchPredictor.from_pretrained("facebook/sam2.1-hiera-base-plus")
        self.transform_mask = ResizeMaskWithAspect(256)

    def forward(self, image, guidance):
        """
        Прямой проход: для батча изображений и боксов-подсказок.
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Здесь передаём guidance в качестве боксов
            feas, _, _, all_low_res_masks = self.predictor.run_batch_torch(image, box_batch=guidance)
        pred = self.m2m(feas, image, all_low_res_masks)
        return pred

    def forward_inference(self, image_dict):
        """
        Инференс с дополнительной постобработкой масок:
          - Обрезка до pad_shape
          - Добавление размерности канала (если необходимо)
          - Интерполяция до оригинального размера
        """
        image = image_dict["image"]
        bbox = image_dict.get("bbox")
        input_size = image_dict["pad_shape"]
        original_size = image_dict["ori_shape"]
        pts_torch = image_dict.get("point")
        labels_torch = image_dict.get("point_labels_batch")
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            feas, all_masks, _, all_low_res_masks = self.predictor.run_batch_torch(
                image,
                box_batch=bbox,
                point_coords_batch=pts_torch,
                point_labels_batch=labels_torch
            )
            # Обрезаем маски до размеров паддинга
            masks = all_masks[..., :input_size[0], :input_size[1]]
            # Если размерность каналов отсутствует — добавляем её
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            if all_low_res_masks.ndim == 3:
                all_low_res_masks = all_low_res_masks.unsqueeze(1)
            # Интерполируем маски до оригинального размера
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pred = self.m2m(feas, image, all_low_res_masks)
        return feas, pred, masks
    
    
    def tresh_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return (mask > self.predictor.mask_threshold).to(torch.float32)
    
    def forward_video(self, image_embed: dict, full_masks: dict, low_masks: dict, width_origin: int, height_origin: int, torch_images: dict):
        
        frames_predict = {}
        
        for key in image_embed:
            img_emb = image_embed[key]
            full_mask = self.tresh_mask(full_masks[key])
            low_mask = self.tresh_mask(low_masks[key])
            torch_image = torch_images[key]["image"]
                        
            masks = F.interpolate(full_mask, (height_origin, width_origin), mode="bilinear", align_corners=False)
            
            # mask_vis = cv2.cvtColor((masks[0].permute(1, 2, 0) * 255).to(dtype=torch.uint8).cpu().numpy(), cv2.COLOR_GRAY2BGR)
            
            # cv2.imwrite("/app/mask.png", mask_vis)
            
            # low_mask = self._prepare_mask(low_mask, height_origin, width_origin)
            
            pred = self.m2m(img_emb, torch_image, low_mask)

            frames_predict[key] = (img_emb, pred, masks)
        
        return frames_predict
    
    
    def _prepare_mask(self, mask: torch.Tensor, height_origin, width_origin):            

        # Применяем resize
        mask_resized = self.transform_mask.apply_mask(mask, height_origin=height_origin, width_origin=width_origin)

        # Паддинг
        h, w = mask_resized.shape[2:]
        padh = self.transform_mask.target_size  - h
        padw = self.transform_mask.target_size - w
        mask_tensor = F.pad(mask_resized, (0, padw, 0, padh))

        # Возвращаем батч [1, 3, H, W]
        return mask_tensor
        
        

def sam2_get_generator_m2m(seg, m2m):
    if 'sam' in seg:
        generator = sam2_m2m(seg=seg, m2m=m2m)
    return generator
