import torch
import torch.nn as nn
from typing import Optional
from networks import m2ms  # предполагается, что m2ms содержит нужные генераторы

import sys
sys.path.insert(0, './segment-anything')
from sam2.torch_sam_video_predictor import SAM2VideoTorchPredictor

class sam2_video_m2m(nn.Module):
    def __init__(self, seg: str, m2m: str):
        super().__init__()
        if m2m not in m2ms.__all__:
            raise NotImplementedError(f"Unknown M2M {m2m}")
        self.m2m = m2ms.__dict__[m2m](nc=256)
        # Инициализируем видео-предиктор-обёртку
        self.predictor = SAM2VideoTorchPredictor.from_pretrained("facebook/sam2.1-hiera-base-plus")

    def forward(self, video: torch.Tensor, guidance: Optional[dict] = None):
        """
        Прямой проход для видео.
        
        Аргументы:
          video (torch.Tensor): видео в формате [T, C, H, W]
          guidance (dict, optional): словарь с промтами для первого кадра. Может содержать ключи:
              - 'box': бокс (например, в формате XYXY)
              - 'point_coords': координаты точек, тензор
              - 'point_labels': метки точек, тензор
        Возвращает:
          pred (torch.Tensor): предсказание m2m, полученное по всему видео (batch по кадрам)
          frame_indices (List[int]): список индексов кадров, для которых рассчитаны выходы
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Инициализируем состояние инференса из видео-тензора через обёртку
            inference_state = self.predictor.init_state_torch(video)
            # Если заданы промты, применяем их к первому кадру
            if guidance is not None:
                if ("box" in guidance) or ("point_coords" in guidance and "point_labels" in guidance):
                    self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=1,
                        points=guidance.get("point_coords"),
                        labels=guidance.get("point_labels"),
                        box=guidance.get("box"),
                    )
            # Запускаем видео-прогон: получаем эмбеддинги и низкоразрешённые маски по кадрам
            feas, low_res_masks, frame_indices = self.predictor.run_video_torch(inference_state)
        # Передаём результаты в модуль m2m
        pred = self.m2m(feas, video, low_res_masks)
        return pred, frame_indices

def sam2_get_generator_video(seg: str, m2m: str) -> nn.Module:
    """
    Фабричный метод для создания генератора видео-сегментации.
    Если в строке seg присутствует 'sam', возвращается объект sam2_video_m2m.
    """
    if 'sam' in seg:
        generator = sam2_video_m2m(seg=seg, m2m=m2m)
    else:
        raise NotImplementedError("Only SAM-based generators are supported for video.")
    return generator
