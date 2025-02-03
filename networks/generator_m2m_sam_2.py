import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.m2ms import SAM_Decoder_Deep
from sam2.modeling.sam2_base import SAM2Base
from sam2.torch_sam_image_predictor import SAM2TorchPredictor
from sam.segment_anything.utils.transforms import ResizeMaskWithAspect


class Sam2M2m(nn.Module):
    def __init__(self, 
                 seg: SAM2Base,
                 device, 
                 low_resolution_mask: int = 256,
                 ):
        """
        Initializes the Sam2M2m module which integrates a SAM-based predictor and an M2M network.

        Args:
            seg: SAM2Base segmentation model.
            low_resolution_mask (int): Size of the low resolution mask.
        """
        super().__init__()
        self.device = device
        self.m2m = SAM_Decoder_Deep()
        self.predictor = SAM2TorchPredictor(seg)
        self.transform_mask = ResizeMaskWithAspect(low_resolution_mask)

    def forward(self, image, guidance):
        """
        Forward pass for a batch of images with provided box prompts.

        Args:
            image (np.ndarray): Input image or batch of images.
            guidance (torch.Tensor): Box prompts for the predictor.

        Returns:
            torch.Tensor: Prediction output from the M2M network.
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features, _, _, all_low_res_masks = self.predictor.run_predict(
                image, box=guidance
            )
        prediction = self.m2m(features, image, all_low_res_masks)
        return prediction

    def forward_inference(self, image_dict: dict):
        """
        Performs inference with additional mask post-processing:
          - Thresholding of masks.
          - Adding a channel dimension if necessary.
          - Interpolating masks to the original image size.

        Args:
            image_dict (dict): Dictionary containing:
                - "image": The original image tensor.
                - "np_image": The numpy image(s) for the predictor.
                - "ori_shape": Original image size (height, width).
                - Optionally, "bbox": Box prompt.
                - Optionally, "point": Point prompts.
                - Optionally, "point_labels_batch": Corresponding point labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - features: Extracted image features.
                - prediction: Output from the M2M network.
                - masks: Post-processed masks resized to the original image size.
        """
        image = image_dict["image"]
        bbox = image_dict.get("bbox")
        original_size = image_dict["ori_shape"]
        pts_torch = image_dict.get("point")
        labels_torch = image_dict.get("point_labels_batch")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features, all_masks, _, all_low_res_masks = self.predictor.run_predict(
                image_dict["np_image"],
                box=bbox,
                point_coords=pts_torch,
                point_labels=labels_torch
            )

            masks = self.threshold_mask(all_masks)
            low_res_masks = self.threshold_mask(all_low_res_masks)

            # Add channel dimension if needed.
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            if low_res_masks.ndim == 3:
                low_res_masks = low_res_masks.unsqueeze(1)

            # Interpolate masks to the original image size.
            masks = F.interpolate(masks, size=original_size, mode="bilinear", align_corners=False)

        prediction = self.m2m(features, image, low_res_masks)
        return features, prediction, masks

    def threshold_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Applies thresholding to the mask using the predictor's mask_threshold.

        Args:
            mask (torch.Tensor): Input mask tensor.

        Returns:
            torch.Tensor: Binary mask tensor (float32) after thresholding.
        """
        return (mask > self.predictor.mask_threshold).to(torch.float32)

    def forward_video(
        self,
        image_embed: dict,
        full_masks: dict,
        low_masks: dict,
        width_origin: int,
        height_origin: int,
        torch_images: dict,
    ) -> dict:
        """
        Performs inference on video frames:
          - Processes each frame's masks.
          - Thresholds and resizes masks to the original resolution.
          - Generates predictions using the M2M network.

        Args:
            image_embed (dict): Dictionary of image embeddings for each frame.
            full_masks (dict): Dictionary of high resolution masks for each frame.
            low_masks (dict): Dictionary of low resolution masks for each frame.
            width_origin (int): Original image width.
            height_origin (int): Original image height.
            torch_images (dict): Dictionary of original image tensors for each frame.

        Returns:
            dict: A dictionary where each key (frame identifier) maps to a tuple:
                  (image embedding, M2M prediction, resized mask).
        """
        frames_predict = {}
        for key, features in image_embed.items():
            full_mask = self.threshold_mask(full_masks[key].to(self.device))
            low_mask = self.threshold_mask(low_masks[key].to(self.device))
            torch_image = torch_images[key]["image"].to(self.device)

            # Interpolate the full mask to the original resolution.
            resized_mask = F.interpolate(
                full_mask, size=(height_origin, width_origin),
                mode="bilinear", align_corners=False
            )
            features = features.to(self.device)
            prediction = self.m2m(features, torch_image, low_mask)
            
            for key_tensor, value in prediction.items():
                prediction[key_tensor] = value.cpu()
            
            frames_predict[key] = (features.cpu(), prediction, resized_mask.cpu())
            
        return frames_predict
