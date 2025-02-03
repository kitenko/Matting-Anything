from typing import Optional, Tuple

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
