import os
import random
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F

# Local imports
import utils
from networks.generator_m2m_sam_2 import Sam2M2m
from sam.segment_anything.utils.transforms import ResizeLongestSide
from sam2.torch_sam_image_predictor import SAM2TorchPredictor

#########################################################
# Configuration (can be moved to a separate config file)
#########################################################
PALETTE_back: Tuple[int, int, int] = (0, 0, 0)  # Color for "green screen" (default black)
BACKGROUND_FOLDER: str = 'assets/backgrounds'
if os.path.exists(BACKGROUND_FOLDER):
    background_list: List[str] = os.listdir(BACKGROUND_FOLDER)
else:
    background_list = []


class MAMInferencer:
    """
    Class for inference of the Matting Anything Model (MAM) without Gradio/GroundingDINO.
    
    The `predict` method accepts:
      - image: np.ndarray of shape (H, W, 3) in RGB uint8 format.
      - points: List of tuples [((x, y), label)] (label is int: 0 or 1).
      - bboxes: List of tuples [((x1, y1), (x2, y2))] (coordinates for top-left and bottom-right).
    and returns a tuple of three images (com_img, green_img, alpha_rgb), all in uint8 RGB format.
    """

    def __init__(
        self,
        model_video: Any,
        mam_checkpoint: str,
        device: str = "cuda",
        image_size: int = 1024
    ) -> None:
        """
        Initialize the MAM model:
          - Loads the generator via sam2_get_generator_m2m.
          - Loads weights from mam_checkpoint.
          - Initializes a ResizeLongestSide transformation.
          
        :param model_video: SAM model for video.
        :param mam_checkpoint: Path to the .pth file with MAM weights.
        :param device: "cuda" or "cpu".
        :param image_size: Target image size for resizing.
        """
        self.device: str = device
        # Initialize the MAM model
        self.mam_model: SAM2TorchPredictor = Sam2M2m(model_video, device=device)
        self.mam_model.to(self.device)

        # Load checkpoint
        checkpoint: Dict[str, Any] = torch.load(mam_checkpoint, map_location=self.device)
        self.mam_model.m2m.load_state_dict(
            utils.remove_prefix_state_dict(checkpoint['state_dict']),
            strict=True
        )
        self.mam_model.eval()

        # Transformation for Segment-Anything
        self.transform: ResizeLongestSide = ResizeLongestSide(image_size)
        self.image_size: int = image_size

        print("[INFO] MAM model successfully loaded and ready.")

    def _prepare_video_frame(self, image_np: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Prepares an input video frame for inference:
          - Converts the image to a tensor and normalizes it.
          - Resizes the image to self.image_size using bilinear interpolation.
          
        :param image_np: Input image as a numpy array.
        :return: Tuple (image_tensor, original_size), where image_tensor has shape [1, 3, H, W].
        """
        original_size: Tuple[int, int] = image_np.shape[:2]

        image_tensor: torch.Tensor = torch.from_numpy(image_np).to(self.device)
        image_tensor = image_tensor.permute(2, 0, 1).float()

        # Normalize
        pixel_mean: torch.Tensor = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(3, 1, 1)
        pixel_std: torch.Tensor = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(3, 1, 1)
        image_tensor = ((image_tensor - pixel_mean) / pixel_std).unsqueeze(0)
        image_tensor = F.interpolate(image_tensor, (self.image_size, self.image_size), mode="bilinear")

        return image_tensor, original_size

    def _prepare_points(self, points: List[Tuple[Tuple[float, float], int]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Converts a list of points into numpy arrays for coordinates and labels.
        
        :param points: List of ((x, y), label).
        :return: Tuple (pts_array, labels_array) or (None, None) if empty.
        """
        if len(points) == 0:
            return None, None

        raw_pts: List[List[float]] = []
        raw_labels: List[int] = []
        for (x, y), label in points:
            raw_pts.append([x, y])
            raw_labels.append(label)

        pts_array: np.ndarray = np.array(raw_pts, dtype=float)
        labels_array: np.ndarray = np.array(raw_labels, dtype=float)
        return pts_array, labels_array

    def _prepare_bboxes(self, bboxes: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Optional[np.ndarray]:
        """
        Converts a list of bounding boxes into a numpy array of shape (M, 4).
        
        :param bboxes: List of ((x1, y1), (x2, y2)).
        :return: Numpy array of shape (M, 4) or None if empty.
        """
        if len(bboxes) == 0:
            return None
        data: List[List[float]] = []
        for (x1, y1), (x2, y2) in bboxes:
            data.append([x1, y1, x2, y2])
        return np.array(data, dtype=float)

    def _resize_alpha_predictions(
        self,
        alpha_os1: torch.Tensor,
        alpha_os4: torch.Tensor,
        alpha_os8: torch.Tensor,
        target_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resizes the alpha prediction tensors to the target shape.
        
        :param alpha_os1: Alpha prediction tensor from OS1.
        :param alpha_os4: Alpha prediction tensor from OS4.
        :param alpha_os8: Alpha prediction tensor from OS8.
        :param target_shape: Target shape (H, W).
        :return: Tuple of resized tensors (alpha_os1, alpha_os4, alpha_os8).
        """
        alpha_os8 = F.interpolate(alpha_os8, target_shape, mode="bilinear", align_corners=False)
        alpha_os4 = F.interpolate(alpha_os4, target_shape, mode="bilinear", align_corners=False)
        alpha_os1 = F.interpolate(alpha_os1, target_shape, mode="bilinear", align_corners=False)
        return alpha_os1, alpha_os4, alpha_os8

    def _apply_guidance(
        self,
        guidance_mode: str,
        post_mask: torch.Tensor,
        alpha_os1: torch.Tensor,
        alpha_os4: torch.Tensor,
        alpha_os8: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies guidance rules to combine alpha predictions.
        
        :param guidance_mode: "mask" or "alpha".
        :param post_mask: The post-processed mask tensor.
        :param alpha_os1: Alpha prediction from OS1.
        :param alpha_os4: Alpha prediction from OS4.
        :param alpha_os8: Alpha prediction from OS8.
        :return: Combined alpha prediction tensor.
        """
        if guidance_mode == 'mask':
            post_mask = post_mask.clone()
            weight_os8: torch.Tensor = utils.get_unknown_tensor_from_mask_oneside(post_mask, rand_width=10, train_mode=False).cpu()
            post_mask[weight_os8 > 0] = alpha_os8[weight_os8 > 0]
            alpha_pred: torch.Tensor = post_mask.clone().detach()
        else:
            weight_os8: torch.Tensor = utils.get_unknown_box_from_mask(post_mask).cpu()
            alpha_os8 = alpha_os8.clone()
            alpha_os8[weight_os8 > 0] = post_mask[weight_os8 > 0]
            alpha_pred = alpha_os8.clone().detach()

        weight_os4: torch.Tensor = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=20, train_mode=False).cpu()
        alpha_pred[weight_os4 > 0] = alpha_os4[weight_os4 > 0]

        weight_os1: torch.Tensor = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=10, train_mode=False).cpu()
        alpha_pred[weight_os1 > 0] = alpha_os1[weight_os1 > 0]

        return alpha_pred

    def _assemble_result(self, alpha_pred: np.ndarray, image: np.ndarray, background_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assembles the final results:
          - Converts alpha_pred to an RGB image.
          - Composites the image with a background if available.
          - Creates a "green" (or black) screen image.
        
        :param alpha_pred: Alpha prediction as a numpy array (H, W) with values in [0, 1].
        :param image: Original image (H, W, 3).
        :param background_type: "real_world_sample" or another value.
        :return: Tuple (com_img, green_img, alpha_rgb).
        """
        alpha_rgb: np.ndarray = cv2.cvtColor(np.uint8(alpha_pred * 255), cv2.COLOR_GRAY2RGB)
        if background_type == 'real_world_sample' and background_list:
            bg_file: str = os.path.join(BACKGROUND_FOLDER, random.choice(background_list))
            background_img: Optional[np.ndarray] = cv2.imread(bg_file)  # BGR
            if background_img is not None:
                background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
                background_img = cv2.resize(background_img, (image.shape[1], image.shape[0]))
                com_img: np.ndarray = alpha_pred[..., None] * image + (1 - alpha_pred[..., None]) * np.uint8(background_img)
                com_img = np.uint8(com_img)
            else:
                com_img = image.copy()
        else:
            com_img = image.copy()

        green_img: np.ndarray = alpha_pred[..., None] * image + (1 - alpha_pred[..., None]) * np.array([PALETTE_back], dtype='uint8')
        green_img = np.uint8(green_img)
        return com_img, green_img, alpha_rgb

    def predict(
        self,
        image: np.ndarray,  # (H, W, 3) RGB uint8
        points: List[Tuple[Tuple[float, float], int]],  # List of ((x, y), label)
        bboxes: List[Tuple[Tuple[float, float], Tuple[float, float]]],  # List of ((x1, y1), (x2, y2))
        guidance_mode: str = "alpha",  # "alpha" or "mask"
        background_type: str = "real_world_sample"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main inference method for a single image:
          1. Prepares the input image (resize+pad).
          2. Converts points and bboxes.
          3. Constructs a sample dictionary for the MAM model.
          4. Runs forward_inference and processes the alpha predictions.
          5. Assembles the result: composited image, green screen, and alpha mask in RGB.
          
        :return: Tuple (com_img, green_img, alpha_rgb) – all (H, W, 3) uint8, RGB.
        """
        # 1) Prepare input image
        image_tensor, original_size = self._prepare_video_frame(image)

        # 2) Prepare points and bboxes
        pts_torch, labels_torch = self._prepare_points(points)
        bboxes_torch: Optional[np.ndarray] = self._prepare_bboxes(bboxes)

        # 3) Construct sample for inference
        sample: Dict[str, Any] = {
            'np_image': image,
            'image': image_tensor,
            'ori_shape': original_size,
        }
        if pts_torch is not None and labels_torch is not None:
            sample['point'] = pts_torch
            sample['point_labels_batch'] = labels_torch
        if bboxes_torch is not None:
            sample['bbox'] = bboxes_torch

        # 4) Run MAM and process predictions
        with torch.no_grad():
            _, pred, post_mask = self.mam_model.forward_inference(sample)

            # Extract alpha predictions at different scales
            alpha_pred_os1: torch.Tensor = pred['alpha_os1']
            alpha_pred_os4: torch.Tensor = pred['alpha_os4']
            alpha_pred_os8: torch.Tensor = pred['alpha_os8']

            # Resize predictions to the original size
            alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = self._resize_alpha_predictions(
                alpha_pred_os1, alpha_pred_os4, alpha_pred_os8, sample['ori_shape']
            )

            # Apply guidance rules
            alpha_pred: torch.Tensor = self._apply_guidance(guidance_mode, post_mask, alpha_pred_os1, alpha_pred_os4, alpha_pred_os8)

            # Convert final alpha mask to numpy (assumes batch size = 1)
            alpha_pred_np: np.ndarray = alpha_pred[0][0].cpu().numpy()  # (H, W) float [0..1]

        # 5) Assemble final result
        return self._assemble_result(alpha_pred_np, image, background_type)

    def predict_video(
        self,
        image_embed: Dict[Any, torch.Tensor],
        full_masks: Dict[Any, torch.Tensor],
        low_masks: Dict[Any, torch.Tensor],
        original_image_np: np.ndarray,
        guidance_mode: str = "alpha",
        background_type: str = "real_world_sample"
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Video inference method. For each frame:
          - Prepares the image tensor.
          - Runs forward_video of the MAM model.
          - Processes the alpha masks (similarly to predict).
          - Assembles the result for each frame.
          
        :return: Dictionary where the key is the frame index and the value is (com_img, green_img, alpha_rgb).
        """
        # Determine original frame dimensions
        height_origin, width_origin = original_image_np.shape[1], original_image_np.shape[2]
        torch_frames: Dict[int, Dict[str, Any]] = {}
        for id_frame, np_image in enumerate(original_image_np):
            image_tensor, original_size = self._prepare_video_frame(np_image)
            torch_frames[id_frame] = {
                'image': image_tensor,
                'ori_shape': original_size,
            }

        result_masks: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # Choose a random background file (same for all frames) if available
        bg_file: Optional[str] = os.path.join(BACKGROUND_FOLDER, random.choice(background_list)) if background_list else None

        with torch.no_grad():
            frames_predict: Dict[Any, Tuple[Any, Dict[str, torch.Tensor], torch.Tensor]] = self.mam_model.forward_video(
                image_embed, full_masks, low_masks, width_origin, height_origin, torch_frames
            )

            for id_frame, (_, pred, post_mask) in frames_predict.items():
                alpha_pred_os1: torch.Tensor = pred['alpha_os1']
                alpha_pred_os4: torch.Tensor = pred['alpha_os4']
                alpha_pred_os8: torch.Tensor = pred['alpha_os8']

                ori_shape: Tuple[int, int] = torch_frames[id_frame]['ori_shape']
                alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = self._resize_alpha_predictions(
                    alpha_pred_os1, alpha_pred_os4, alpha_pred_os8, ori_shape
                )

                alpha_pred: torch.Tensor = self._apply_guidance(guidance_mode, post_mask, alpha_pred_os1, alpha_pred_os4, alpha_pred_os8)
                alpha_pred_np: np.ndarray = alpha_pred[0][0].cpu().numpy()  # (H, W) float [0..1]

                # Assemble result for the frame
                if background_type == 'real_world_sample' and bg_file is not None:
                    background_img: Optional[np.ndarray] = cv2.imread(bg_file)  # BGR
                    if background_img is not None:
                        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
                        background_img = cv2.resize(background_img, (original_image_np[id_frame].shape[1], original_image_np[id_frame].shape[0]))
                        com_img: np.ndarray = alpha_pred_np[..., None] * original_image_np[id_frame] + (1 - alpha_pred_np[..., None]) * np.uint8(background_img)
                        com_img = np.uint8(com_img)
                    else:
                        com_img = original_image_np[id_frame].copy()
                else:
                    com_img = original_image_np[id_frame].copy()

                result_masks[id_frame] = com_img

        return result_masks
