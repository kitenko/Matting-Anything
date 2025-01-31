# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from   utils import CONFIG
from   networks import m2ms, ops
import sys
sys.path.insert(0, './segment-anything')
from sam2.sam2_image_predictor import SAM2ImagePredictor

class sam2_m2m(nn.Module):
    def __init__(self, seg, m2m):
        super().__init__()
        if m2m not in m2ms.__all__:
            raise NotImplementedError("Unknown M2M {}".format(m2m))
        self.m2m = m2ms.__dict__[m2m](nc=256)
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-base-plus")


    def forward(self, image, guidance):
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image_batch(image)
            _, _, all_low_res_masks_tensor = self.predictor.predict_batch(box_batch=guidance)
            feas = self.predictor._features["image_embed"]
            # feas, masks = self.predictor.model.forward_m2m(image, guidance)
        # with torch.no_grad():
        #     feas, masks = self.seg_model.forward_m2m(image, guidance, multimask_output=True)
        pred = self.m2m(feas, image, all_low_res_masks_tensor)
        return pred
    
    
    def forward_inference(self, image_dict):
        image = image_dict["image"]
        bbox = image_dict.get("bbox")
        
        input_size=image_dict["pad_shape"]
        original_size=image_dict["ori_shape"]
        
        pts_torch = image_dict.get('point')
        labels_torch = image_dict.get('point_labels_batch')
        
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image_batch(image)
            all_masks, _, all_low_res_masks_tensor = self.predictor.predict_batch(box_batch=bbox, point_coords_batch=pts_torch, point_labels_batch=labels_torch)
            feas = self.predictor._features["image_embed"]
            masks = all_masks[..., : input_size[0], : input_size[1]]
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
            # feas, masks = self.predictor.model.forward_m2m(image, guidance)
        # with torch.no_grad():
        #     feas, masks = self.seg_model.forward_m2m(image, guidance, multimask_output=True)
        pred = self.m2m(feas, image, all_low_res_masks_tensor)
        return feas, pred, masks


    # def forward_inference(self, image_dict):
    #     self.seg_model.eval()
    #     with torch.no_grad():
    #         feas, masks, post_masks = self.seg_model.forward_m2m_inference(image_dict, multimask_output=True)
    #     pred = self.m2m(feas, image_dict["image"], masks)
    #     return feas, pred, post_masks

def sam2_get_generator_m2m(seg, m2m):
    if 'sam' in seg:
        generator = sam2_m2m(seg=seg, m2m=m2m)
    return generator