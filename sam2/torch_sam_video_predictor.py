import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
from typing import Optional, Tuple

class SAM2VideoTorchPredictor:
    """
    Обёртка для SAM2VideoPredictor, позволяющая работать с видео-тензорами.
    Предоставляет методы init_state_torch, run_video_torch, get_image_embedding_from_state,
    а также форвардер для add_new_points_or_box и forward.
    """
    def __init__(self, sam_model: SAM2VideoPredictor):
        self.model = sam_model
        self.device = sam_model.device
        self.image_size = sam_model.image_size
        try:
            self._bb_feat_sizes = sam_model._bb_feat_sizes
        except AttributeError:
            # Если базовая модель не определяет этот атрибут, задаем значение по умолчанию.
            self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2VideoTorchPredictor":
        from sam2.build_sam import build_sam2_video_predictor_hf
        sam_model = build_sam2_video_predictor_hf(model_id, **kwargs)
        instance = cls(sam_model)
        return instance

    @torch.inference_mode()
    def init_state_torch(
        self,
        video_tensor: torch.Tensor,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict:
        """
        Инициализирует состояние инференса из видео-тензора [T, C, H, W].
        """
        T, C, H, W = video_tensor.shape
        device = self.device
        images = [video_tensor[i] for i in range(T)]
        inference_state = {
            "images": images,
            "num_frames": T,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "video_height": H,
            "video_width": W,
            "device": device,
            "storage_device": torch.device("cpu") if offload_state_to_cpu else device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": {},
            "obj_idx_to_id": {},
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {}
        }
        # Прогрев: вычисляем фичи для первого кадра
        self.model._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    @torch.inference_mode()
    def run_video_torch(
        self,
        inference_state: dict,
        start_frame_idx: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Прогоняет видео по кадрам и собирает результаты.
        Возвращает:
          - image_embeds: [T, C, H_feat, W_feat]
          - low_res_masks: [T, 1, H_lr, W_lr]
          - frame_indices: список обработанных индексов кадров.
        """
        features_list = []
        low_res_masks_list = []
        frame_indices = []
        for frame_idx, obj_ids, video_res_masks in self.model.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse
        ):
            frame_indices.append(frame_idx)
            image_embed = self.get_image_embedding_from_state(inference_state, frame_idx)
            features_list.append(image_embed)  # image_embed имеет форму [1, C, H_feat, W_feat]
            if video_res_masks.ndim == 3:
                video_res_masks = video_res_masks.unsqueeze(1)
            low_res = torch.nn.functional.interpolate(video_res_masks,
                                                      size=(self.image_size, self.image_size),
                                                      mode="bilinear",
                                                      align_corners=False)
            low_res_masks_list.append(low_res)
        image_embeds = torch.cat(features_list, dim=0)
        low_res_masks = torch.cat(low_res_masks_list, dim=0)
        return image_embeds, low_res_masks, frame_indices

    def get_image_embedding_from_state(
        self, inference_state: dict, frame_idx: int, batch_size: int = 1
    ) -> torch.Tensor:
        """
        Возвращает эмбеддинг изображения для указанного кадра.
        """
        if frame_idx in inference_state["cached_features"]:
            image, backbone_out = inference_state["cached_features"][frame_idx]
            _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, feat_size[0], feat_size[1])
                for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
            ][::-1]
            return feats[-1]
        else:
            self.model._get_image_feature(inference_state, frame_idx, batch_size)
            return self.get_image_embedding_from_state(inference_state, frame_idx, batch_size)
    
    # --- Добавляем форвардеры для методов, которые требуются при работе с интерактивным видео ---
    def add_new_points_or_box(self, *args, **kwargs):
        return self.model.add_new_points_or_box(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
