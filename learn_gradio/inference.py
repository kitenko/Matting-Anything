import os
import random
import cv2
import numpy as np
import torch
from torch.nn import functional as F

# Локальные импорты
import utils
from networks.generator_m2m_sam_2 import sam2_get_generator_m2m
from segment_anything.utils.transforms import ResizeLongestSide
from sam2.torch_sam_image_predictor import SAM2TorchPredictor

#########################################################
# Настройки (при необходимости можно вынести в конфиг)
#########################################################
PALETTE_back = (0, 0, 0)  # цвет для "зелёного экрана", по умолчанию чёрный
BACKGROUND_FOLDER = 'assets/backgrounds'
if os.path.exists(BACKGROUND_FOLDER):
    background_list = os.listdir(BACKGROUND_FOLDER)
else:
    background_list = []


class MAMInferencer:
    """
    Класс для инференса MAM (Matting Anything Model) без Gradio/GroundingDINO.

    Метод `predict` принимает:
      - image (H, W, 3) в RGB
      - points: List[Tuple[(x, y), label]]  (label = int (0 или 1))
      - bboxes: List[Tuple[(x1, y1), (x2, y2)]] (левая верхняя и правая нижняя точки)
    и возвращает кортеж из трёх изображений (com_img, green_img, alpha_rgb).
    """

    def __init__(
        self,
        mam_checkpoint: str,
        device: str = "cuda"
    ):
        """
        Инициализация MAM-модели:
          - Загружаем модель `sam2_get_generator_m2m`.
          - Применяем weights из mam_checkpoint.
          - Готовим трансформацию ResizeLongestSide(1024).

        :param mam_checkpoint: Путь к .pth-файлу с весами MAM
        :param device: "cuda" или "cpu"
        """
        self.device = device
        # Инициализация MAM
        self.mam_model: SAM2TorchPredictor = sam2_get_generator_m2m(seg='sam_vit_b', m2m='sam_decoder_deep')
        self.mam_model.to(self.device)

        # Загрузка чекпоинта
        checkpoint = torch.load(mam_checkpoint, map_location=self.device)
        self.mam_model.m2m.load_state_dict(
            utils.remove_prefix_state_dict(checkpoint['state_dict']),
            strict=True
        )
        self.mam_model.eval()

        # Трансформация для Segment-Anything
        self.transform = ResizeLongestSide(1024)

        print("[INFO] MAM-модель успешно загружена и готова к работе.")

    def _prepare_image(self, image_np: np.ndarray):
        """
        Предобработка входного изображения:
          1. ResizeLongestSide
          2. torch.as_tensor + нормализация
          3. Padding до (1024, 1024)

        :param image_np: (H, W, 3) uint8 (RGB)
        :return:
            - image: тензор [1, 3, H_pad, W_pad] (на self.device)
            - original_size: (H, W) исходного изображения
            - pad_shape: (h_after_pad, w_after_pad) до 1024
        """
        # Оригинальный размер
        original_size = image_np.shape[:2]

        # Применяем resize
        image_resized = self.transform.apply_image(image_np)
        image_tensor = torch.as_tensor(image_resized, device=self.device)
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()  # (3, H_res, W_res)

        # Нормализация
        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(3,1,1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(3,1,1)
        image_tensor = (image_tensor - pixel_mean) / pixel_std

        # Паддинг
        h, w = image_tensor.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        image_tensor = F.pad(image_tensor, (0, padw, 0, padh))  # (3, 1024, 1024)

        # Возвращаем батч [1, 3, H, W]
        return image_tensor.unsqueeze(0), original_size, (h, w)
    
    
    def _prepare_video_frame(self, image_np: np.ndarray):
        # Оригинальный размер
        original_size = image_np.shape[:2]


        image_tensor = torch.from_numpy(image_np).to(self.device)
        image_tensor = image_tensor.permute(2, 0, 1).float()

        # Нормализация
        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(3,1,1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(3,1,1)
        image_tensor = ((image_tensor - pixel_mean) / pixel_std).unsqueeze(0)
        
        image_tensor = F.interpolate(image_tensor, (1024, 1024), mode="bilinear")

        # Возвращаем батч [1, 3, H, W]
        return image_tensor, original_size

    def _prepare_points(self, points, original_size):
        """
        Подготовка списка точек:
          - points: List[Tuple[(x, y), label]],
                    где (x, y) — координаты, label — int (0 или 1)
          - Преобразуем их к (y, x) и потом адаптируем к ресайзу (apply_coords).
          - Формируем тензоры (1, N, 2) для 'point'
            и (1, N) для 'point_labels_batch'.
        """
        if len(points) == 0:
            return None, None

        raw_pts = []
        raw_labels = []
        for (x, y), label in points:
            raw_pts.append([x, y])
            raw_labels.append(label)  # int(0 или 1)

        pts_array = np.array(raw_pts, dtype=float)   # (N, 2) (x, y)
        labels_array = np.array(raw_labels, dtype=float)  # (N,)

        # Нужно swap: (x, y) -> (y, x), т.к. apply_coords ожидает (y, x)
        pts_array_swapped = pts_array[:, [1, 0]]
        # Применяем SAM-трансформацию
        pts_array_swapped = self.transform.apply_coords(pts_array_swapped, original_size)
        
        # Приводим к тензору
        pts_torch = torch.tensor(pts_array_swapped, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        labels_torch = torch.tensor(labels_array, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        return pts_torch, labels_torch

    def _prepare_bboxes(self, bboxes, original_size):
        """
        Подготовка списка bounding boxes:
          - bboxes: List[ ( (x1, y1), (x2, y2) ), ...]
            где (x1,y1) - верхний левый угол, (x2,y2) - правый нижний
          - Преобразуем их для apply_boxes -> (x1, y1, x2, y2)
        """
        if len(bboxes) == 0:
            return None
        data = []
        for (x1, y1), (x2, y2) in bboxes:
            data.append([x1, y1, x2, y2])
        bboxes_array = np.array(data, dtype=float)  # (M, 4)

        # Применяем SAM-трансформацию
        bboxes_array_t = self.transform.apply_boxes(bboxes_array, original_size)
        bboxes_torch = torch.tensor(bboxes_array_t, dtype=torch.float, device=self.device).unsqueeze(0)  # (1, M, 4)
        return bboxes_torch

    def predict(
        self,
        image: np.ndarray,               # (H, W, 3) RGB uint8
        points: list,                    # List[Tuple[(x, y), label]] 
        bboxes: list,                    # List[Tuple[(x1, y1), (x2, y2)]]
        guidance_mode: str = "alpha",    # "alpha" или "mask"
        background_type: str = "real_world_sample"
    ):
        """
        Основной метод инференса:
          1. Предобработка (resize+pad) входного изображения.
          2. Преобразование points и bboxes к новым координатам.
          3. Формирование sample (dict) для MAM-модели.
          4. forward_inference -> альфа-карта.
          5. Сборка результата:
             (a) Подстановка случайного фона (или исходное при отсутствии фонов).
             (b) "Зелёный" (или иной) экран (по умолчанию чёрный).
             (c) Альфа-маска (тоже в rgb-формате).

        :return: (com_img, green_img, alpha_rgb) - все (H, W, 3), uint8, RGB.
        """
        # 1) Подготовка входного изображения
        image_tensor, original_size, pad_size = self._prepare_image(image)

        # 2) Преобразование points и bboxes
        pts_torch, labels_torch = self._prepare_points(points, original_size)
        bboxes_torch = self._prepare_bboxes(bboxes, original_size)

        # 3) Формируем sample
        sample = {
            'image': image_tensor,
            'ori_shape': original_size,
            'pad_shape': pad_size
        }
        if pts_torch is not None and labels_torch is not None:
            sample['point'] = pts_torch
            sample['point_labels_batch'] = labels_torch
        if bboxes_torch is not None:
            sample['bbox'] = bboxes_torch

        # 4) Запуск MAM
        with torch.no_grad():
            _, pred, post_mask = self.mam_model.forward_inference(sample)

            alpha_pred_os1 = pred['alpha_os1']
            alpha_pred_os4 = pred['alpha_os4']
            alpha_pred_os8 = pred['alpha_os8']

            # Обрезаем паддинг
            h_pad, w_pad = pad_size
            alpha_pred_os8 = alpha_pred_os8[..., :h_pad, :w_pad]
            alpha_pred_os4 = alpha_pred_os4[..., :h_pad, :w_pad]
            alpha_pred_os1 = alpha_pred_os1[..., :h_pad, :w_pad]

            # Маштабируем к original_size
            alpha_pred_os8 = F.interpolate(
                alpha_pred_os8, sample['ori_shape'], mode="bilinear", align_corners=False
            )
            alpha_pred_os4 = F.interpolate(
                alpha_pred_os4, sample['ori_shape'], mode="bilinear", align_corners=False
            )
            alpha_pred_os1 = F.interpolate(
                alpha_pred_os1, sample['ori_shape'], mode="bilinear", align_corners=False
            )

            # Guidance
            if guidance_mode == 'mask':
                # Основано на post_mask
                weight_os8 = utils.get_unknown_tensor_from_mask_oneside(
                    post_mask, rand_width=10, train_mode=False
                )
                post_mask[weight_os8 > 0] = alpha_pred_os8[weight_os8 > 0]
                alpha_pred = post_mask.clone().detach()
            else:
                # alpha
                weight_os8 = utils.get_unknown_box_from_mask(post_mask)
                alpha_pred_os8[weight_os8 > 0] = post_mask[weight_os8 > 0]
                alpha_pred = alpha_pred_os8.clone().detach()

            weight_os4 = utils.get_unknown_tensor_from_pred_oneside(
                alpha_pred, rand_width=20, train_mode=False
            )
            alpha_pred[weight_os4 > 0] = alpha_pred_os4[weight_os4 > 0]

            weight_os1 = utils.get_unknown_tensor_from_pred_oneside(
                alpha_pred, rand_width=10, train_mode=False
            )
            alpha_pred[weight_os1 > 0] = alpha_pred_os1[weight_os1 > 0]

            alpha_pred = alpha_pred[0][0].cpu().numpy()  # (H, W) float [0..1]

        # 5) Сборка результата
        alpha_rgb = cv2.cvtColor(np.uint8(alpha_pred * 255), cv2.COLOR_GRAY2RGB)

        # Подстановка фона (real_world_sample) при наличии
        if background_type == 'real_world_sample' and background_list:
            bg_file = os.path.join(BACKGROUND_FOLDER, random.choice(background_list))
            background_img = cv2.imread(bg_file)  # BGR
            if background_img is not None:
                background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
                background_img = cv2.resize(background_img, (image.shape[1], image.shape[0]))
                com_img = alpha_pred[..., None] * image + (1 - alpha_pred[..., None]) * np.uint8(background_img)
                com_img = np.uint8(com_img)
            else:
                com_img = image.copy()
        else:
            # Если нет фонов или иные параметры, оставим исходное
            com_img = image.copy()

        # "Зелёный" (или чёрный) экран
        green_img = alpha_pred[..., None] * image + (1 - alpha_pred[..., None]) * np.array([PALETTE_back], dtype='uint8')
        green_img = np.uint8(green_img)

        return com_img, green_img, alpha_rgb
    
    
    def predict_video(self, 
                      image_embed: dict, 
                      full_masks: dict, 
                      low_masks: dict, 
                      original_image_np: np.ndarray, 
                      guidance_mode: str = "alpha",
                      background_type: str = "real_world_sample"
                      ):
        
        height_origin, width_origin = original_image_np.shape[1], original_image_np.shape[2]
        
        torch_frames = {}
        
        for id_frame, np_image in enumerate(original_image_np):
            image_tensor, original_size = self._prepare_video_frame(np_image)
            
            torch_frames[id_frame] = {
                    'image': image_tensor,
                    'ori_shape': original_size,
                }

        
        result_masks = {}
        
        bg_file = os.path.join(BACKGROUND_FOLDER, random.choice(background_list))
        
        with torch.no_grad():
            frames_predict: dict = self.mam_model.forward_video(image_embed, full_masks, low_masks, width_origin, height_origin, torch_frames)
            
            for id_frame, (img_emb, pred, post_mask) in frames_predict.items():

                alpha_pred_os1 = pred['alpha_os1']
                alpha_pred_os4 = pred['alpha_os4']
                alpha_pred_os8 = pred['alpha_os8']

                # # Обрезаем паддинг
                # h_pad, w_pad = pad_size
                # alpha_pred_os8 = alpha_pred_os8[..., :h_pad, :w_pad]
                # alpha_pred_os4 = alpha_pred_os4[..., :h_pad, :w_pad]
                # alpha_pred_os1 = alpha_pred_os1[..., :h_pad, :w_pad]

                # Маштабируем к original_size
                alpha_pred_os8 = F.interpolate(
                    alpha_pred_os8, torch_frames[id_frame]['ori_shape'], mode="bilinear", align_corners=False
                )
                alpha_pred_os4 = F.interpolate(
                    alpha_pred_os4, torch_frames[id_frame]['ori_shape'], mode="bilinear", align_corners=False
                )
                alpha_pred_os1 = F.interpolate(
                    alpha_pred_os1, torch_frames[id_frame]['ori_shape'], mode="bilinear", align_corners=False
                )

                # Guidance
                if guidance_mode == 'mask':
                    # Основано на post_mask
                    weight_os8 = utils.get_unknown_tensor_from_mask_oneside(
                        post_mask, rand_width=10, train_mode=False
                    )
                    post_mask[weight_os8 > 0] = alpha_pred_os8[weight_os8 > 0]
                    alpha_pred = post_mask.clone().detach()
                else:
                    # alpha
                    weight_os8 = utils.get_unknown_box_from_mask(post_mask)
                    alpha_pred_os8[weight_os8 > 0] = post_mask[weight_os8 > 0]
                    alpha_pred = alpha_pred_os8.clone().detach()

                weight_os4 = utils.get_unknown_tensor_from_pred_oneside(
                    alpha_pred, rand_width=20, train_mode=False
                )
                alpha_pred[weight_os4 > 0] = alpha_pred_os4[weight_os4 > 0]

                weight_os1 = utils.get_unknown_tensor_from_pred_oneside(
                    alpha_pred, rand_width=10, train_mode=False
                )
                alpha_pred[weight_os1 > 0] = alpha_pred_os1[weight_os1 > 0]

                alpha_pred = alpha_pred[0][0].cpu().numpy()  # (H, W) float [0..1]

                # 5) Сборка результата
                alpha_rgb = cv2.cvtColor(np.uint8(alpha_pred * 255), cv2.COLOR_GRAY2RGB)

                # Подстановка фона (real_world_sample) при наличии
                if background_type == 'real_world_sample' and background_list:
                    background_img = cv2.imread(bg_file)  # BGR
                    if background_img is not None:
                        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
                        background_img = cv2.resize(background_img, (original_image_np[id_frame].shape[1], original_image_np[id_frame].shape[0]))
                        com_img = alpha_pred[..., None] * original_image_np[id_frame] + (1 - alpha_pred[..., None]) * np.uint8(background_img)
                        com_img = np.uint8(com_img)
                    else:
                        com_img = original_image_np[id_frame].copy()
                else:
                    # Если нет фонов или иные параметры, оставим исходное
                    com_img = original_image_np[id_frame].copy()

                # "Зелёный" (или чёрный) экран
                green_img = alpha_pred[..., None] * original_image_np[id_frame] + (1 - alpha_pred[..., None]) * np.array([PALETTE_back], dtype='uint8')
                green_img = np.uint8(green_img)
                
                result_masks[id_frame] = (com_img, green_img, alpha_rgb)

        return result_masks
