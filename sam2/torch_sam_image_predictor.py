from typing import Optional, Tuple, List, Union
import logging

import torch
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2TorchPredictor(SAM2ImagePredictor):
    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        box: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
        normalize_coords: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts a mask for a previously set image using prompts provided as torch.Tensors.
        
        The method selects a single best mask based on the highest IoU prediction (argmax over iou_predictions).

        Args:
            point_coords (Optional[torch.Tensor]): Tensor of point coordinates as prompts.
            point_labels (Optional[torch.Tensor]): Tensor of point labels corresponding to the coordinates.
            box (Optional[torch.Tensor]): Tensor representing a box prompt.
            mask_input (Optional[torch.Tensor]): Optional low-resolution mask input from a previous prediction.
            multimask_output (bool): If True, allows the predictor to generate multiple mask candidates.
            return_logits (bool): If True, returns the raw mask logits instead of binary masks.
            normalize_coords (bool): If True, normalize the point coordinates to the expected range.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - masks: A tensor containing the best predicted mask (binary or logits depending on return_logits).
                - iou_predictions: A tensor with IoU predictions for each mask candidate.
                - low_res_masks: A tensor containing the corresponding low-resolution mask output.
                
        Raises:
            RuntimeError: If no image has been set (use set_image_torch() first).
        """
        if not self._is_image_set:
            raise RuntimeError("Please set the image first using set_image_torch().")
        
        # Preprocess the prompts and convert them into the expected format.
        mask_input_proc, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )
        
        # Perform the prediction.
        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input_proc,
            multimask_output,
            return_logits=return_logits,
        )
        
        # Select the best mask based on the highest IoU prediction.
        best_idx = torch.argmax(iou_predictions, dim=1)
        best_idx_int = best_idx[0]
        masks = masks[0, best_idx_int, :, :].unsqueeze(0)
        low_res_masks = low_res_masks[0, best_idx_int, :, :].unsqueeze(0)
        
        return masks, iou_predictions, low_res_masks

    @torch.no_grad()
    def run_predict(
        self,
        images: np.ndarray,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the complete prediction pipeline on the provided image(s).

        This method sets the image, performs prediction, and returns the image embeddings
        along with the predicted masks, IoU predictions, and low-resolution masks.

        Args:
            images (np.ndarray): The input image(s) in numpy array format.
            **kwargs: Additional keyword arguments passed to `predict_torch()`, such as:
                - point_coords
                - point_labels
                - box
                - mask_input
                - multimask_output
                - return_logits
                - normalize_coords

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - image_embed: The image embedding tensor obtained from the predictor.
                - masks: The predicted mask tensor.
                - iou_predictions: The IoU predictions tensor for the mask candidates.
                - low_res_masks: The low-resolution mask tensor.
        """
        # Set the image (the underlying set_image() method will prepare the image embedding).
        self.set_image(images)
        
        # Run the mask prediction.
        masks, iou_predictions, low_res_masks = self.predict_torch(**kwargs,)
        
        # Retrieve the image embedding from the stored features.
        image_embed = self._features["image_embed"]
        return image_embed, masks, iou_predictions, low_res_masks
    
    @torch.no_grad()
    def torch_set_image_batch(
        self,
        img_batch: torch.Tensor
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
        img_batch (List[Union[torch.Tensor, np.ndarray]]): The input images to embed in RGB format.
            If provided as np.ndarray, images are expected to be in HWC format with pixel values in [0, 255].
        """
        self.reset_predictor()
        self._orig_hw = []
        # Если изображения переданы как numpy, можно сразу записывать их размеры.
        for image in img_batch:
            # image.shape работает как для np.ndarray, так и для torch.Tensor (если оно имеет .shape)
            self._orig_hw.append(image.shape[-2:])
    
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert len(img_batch.shape) == 4 and img_batch.shape[1] == 3, \
            f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Если модель должна добавлять no_mem_embed, делаем это:
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def torch_predict_batch(
        self,
        point_coords_batch: List[torch.Tensor] = None,
        point_labels_batch: List[torch.Tensor] = None,
        box_batch: torch.Tensor = None,
        mask_input_batch: List[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        This function is used in batched mode when the model is expected to generate predictions on multiple images.
        It returns a tuple of lists of masks, IoU predictions, and low-resolution masks.
        All data is handled as torch.Tensors, so no numpy conversions are performed here.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
                - all_masks: List of predicted masks for each image.
                - all_ious: List of predicted IoU values for each image.
                - all_low_res_masks: List of low-resolution mask logits for each image.
        """
        assert self._is_batch, "This function should only be used in batched mode."
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image_batch(...) before mask prediction.")

        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []

        for img_idx in range(num_images):
            # Извлекаем входные подсказки для каждого изображения (предполагается, что они уже в виде torch.Tensor)
            point_coords = point_coords_batch[img_idx] if point_coords_batch is not None else None
            point_labels = point_labels_batch[img_idx] if point_labels_batch is not None else None
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = mask_input_batch[img_idx] if mask_input_batch is not None else None

            # Подготавливаем подсказки; предполагается, что _prep_prompts теперь работает с torch.Tensor
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            # Получаем предсказания для одного изображения.
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            # Сохраняем результаты в виде torch.Tensor (без преобразования в numpy)
            all_masks.append(masks)
            all_ious.append(iou_predictions)
            all_low_res_masks.append(low_res_masks)
            
        all_masks = torch.concatenate(all_masks, dim=0)
        all_ious = torch.concatenate(all_ious, dim=0)
        all_low_res_masks = torch.concatenate(all_low_res_masks, dim=0)

        return all_masks, all_ious, all_low_res_masks
    
    
    def train_step(self, images_batch: torch.Tensor, bbox: torch.Tensor):
        self.torch_set_image_batch(images_batch)
        masks, iou_predictions, low_res_masks = self.torch_predict_batch(box_batch=bbox, multimask_output=False)
        
        image_embed = self._features["image_embed"]
        return image_embed, masks, iou_predictions, low_res_masks
        
