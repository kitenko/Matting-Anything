import logging
from typing import List, Optional, Tuple

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2TorchPredictor(SAM2ImagePredictor):
    """
    Класс SAM2TorchPredictor наследуется от SAM2ImagePredictor и расширяет его новыми методами,
    позволяющими работать с входными данными в формате torch.Tensor и возвращающими torch.Tensor.
    
    Новые методы:
      - set_image_torch: принимает изображение в формате [1, C, H, W]
      - set_image_batch_torch: принимает батч изображений в формате [B, C, H, W]
      - predict_torch: предсказывает маску для одного изображения с torch-промптами
      - predict_batch_torch: предсказывает маски для батча изображений с torch-промптами
      - run_batch_torch: устанавливает батч изображений и выполняет предсказание, возвращая эмбеддинги и результаты
    """

    @torch.no_grad()
    def set_image_torch(self, image: torch.Tensor) -> None:
        """
        Устанавливает изображение для предсказания, принимая его в виде torch.Tensor.
        Ожидается, что image имеет форму [1, C, H, W].
        """
        self.reset_predictor()
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Ожидается изображение в формате [1, C, H, W].")
        image = image.to(self.device)
        # Сохраняем оригинальный размер изображения (H, W)
        self._orig_hw = [tuple(image.shape[2:])]
        logging.info("Вычисление эмбеддингов для переданного torch-изображения...")
        backbone_out = self.model.forward_image(image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        batch_size = 1
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Эмбеддинги изображения вычислены.")

    @torch.no_grad()
    def set_image_batch_torch(self, images: torch.Tensor) -> None:
        """
        Устанавливает батч изображений для предсказания, принимая их в виде torch.Tensor.
        Ожидается, что images имеет форму [B, C, H, W].
        """
        self.reset_predictor()
        if images.ndim != 4:
            raise ValueError("Ожидается батч изображений в формате [B, C, H, W].")
        images = images.to(self.device)
        batch_size = images.shape[0]
        # Для каждого изображения сохраняем оригинальный размер (H, W)
        self._orig_hw = [tuple(images.shape[2:]) for _ in range(batch_size)]
        logging.info("Вычисление эмбеддингов для батча torch-изображений...")
        backbone_out = self.model.forward_image(images)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Эмбеддинги батча изображений вычислены.")

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        box: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Предсказывает маску для ранее установленного изображения, используя промпты в формате torch.Tensor.
        Выбирается одна лучшая маска (по argmax iou_predictions).
        """
        if not self._is_image_set:
            raise RuntimeError("Сначала установите изображение с помощью set_image_torch().")
        
        mask_input_proc, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )
        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input_proc,
            multimask_output,
            return_logits=return_logits,
        )
        
        if not return_logits:
            # Извлекаем лучшую маску
            best_idx = torch.argmax(iou_predictions, dim=1)  # [1]
            best_idx_int = best_idx.item()
            best_mask = masks[0, best_idx_int, :, :].unsqueeze(0)  # [1, H, W]
            best_low_res = low_res_masks[0, best_idx_int, :, :].unsqueeze(0)
            best_mask = (best_mask > self.mask_threshold).to(torch.float32)
            best_low_res = (best_low_res > self.mask_threshold).to(torch.float32)
            masks = best_mask
            low_res_masks = best_low_res
        
        return masks, iou_predictions, low_res_masks

    @torch.no_grad()
    def predict_batch_torch(
        self,
        point_coords_batch: Optional[List[torch.Tensor]] = None,
        point_labels_batch: Optional[List[torch.Tensor]] = None,
        box_batch: Optional[List[torch.Tensor]] = None,
        mask_input_batch: Optional[List[torch.Tensor]] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Предсказывает маски для батча изображений, используя промпты в формате torch.Tensor.
        Для каждого изображения выбирается лучшая маска.
        
        Возвращает:
            masks: [B, 1, H, W]
            iou_predictions: [B, C]
            low_res_masks: [B, 1, H_lr, W_lr]
        """
        if not self._is_image_set or not self._is_batch:
            raise RuntimeError("Сначала установите батч изображений с помощью set_image_batch_torch().")
        
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_iou = []
        all_low_res = []
        
        for img_idx in range(num_images):
            point_coords = point_coords_batch[img_idx] if point_coords_batch is not None else None
            point_labels = point_labels_batch[img_idx] if point_labels_batch is not None else None
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input_proc = mask_input_batch[img_idx] if mask_input_batch is not None else None

            mask_input_proc, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords, point_labels, box, mask_input_proc, normalize_coords, img_idx=img_idx
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input_proc,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            if not return_logits:
                best_idx = torch.argmax(iou_predictions, dim=1)
                best_idx_int = best_idx.item()
                best_mask = masks[0, best_idx_int, :, :].unsqueeze(0)
                best_low_res = low_res_masks[0, best_idx_int, :, :].unsqueeze(0)
                best_mask = (best_mask > self.mask_threshold).to(torch.float32)
                best_low_res = (best_low_res > self.mask_threshold).to(torch.float32)
                masks = best_mask
                low_res_masks = best_low_res
            all_masks.append(masks)
            all_iou.append(iou_predictions)
            all_low_res.append(low_res_masks)
        
        masks_tensor = torch.cat(all_masks, dim=0)      # [B, 1, H, W]
        iou_tensor = torch.cat(all_iou, dim=0)            # [B, C]
        low_res_tensor = torch.cat(all_low_res, dim=0)      # [B, 1, H_lr, W_lr]
        
        return masks_tensor, iou_tensor, low_res_tensor

    @torch.no_grad()
    def run_batch_torch(
        self,
        images: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Устанавливает батч изображений и выполняет предсказание с дополнительными промптами.
        
        Возвращает:
            - image_embed: эмбеддинги изображения [B, C, H_feat, W_feat]
            - masks: предсказанные маски [B, 1, H, W]
            - iou_predictions: оценки качества [B, C]
            - low_res_masks: низкоразрешённые маски [B, 1, H_lr, W_lr]
        """
        self.set_image_batch_torch(images)
        masks, iou_predictions, low_res_masks = self.predict_batch_torch(**kwargs)
        image_embed = self._features["image_embed"]
        return image_embed, masks, iou_predictions, low_res_masks
