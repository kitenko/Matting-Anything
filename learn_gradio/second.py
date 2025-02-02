import cv2
import numpy as np
import gradio as gr
import tempfile
from typing import List, Tuple, Optional

from moviepy import ImageSequenceClip

# Импортируем ваш класс с методом `predict`
import sys
sys.path.append("/app")
from learn_gradio.inference import MAMInferencer
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor_hf

# Создаём экземпляры моделей
model_image = MAMInferencer("/app/checkpoints/return_cousine_pre_train_grad_true_new_shedule_real_world_aug_full_data_sam_2_multiple_mask_True/model_step_30000.pth")
model_video_sam2: SAM2VideoPredictor = build_sam2_video_predictor_hf("facebook/sam2.1-hiera-base-plus")

# Определяем типы данных
Point = Tuple[int, int]   # Координаты точки (x, y)
BBox = Tuple[Point, Point] # Bounding Box (верхний левый и нижний правый угол)

# Цвета и маркеры
COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # Красный (foreground), Синий (background)
MARKERS = [1, 5, 3]
BBOX_COLOR = (0, 255, 0)  # Зеленый

# -------------- Функции для работы с изображениями --------------

def store_img(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[Point, int]], List[BBox], Optional[Point]]:
    return img, img.copy(), [], [], None

def draw_points(img: np.ndarray, points: List[Tuple[Point, int]]) -> np.ndarray:
    img = img.copy()
    for point, label in points:
        cv2.drawMarker(img, point, COLORS[label], markerType=MARKERS[label], markerSize=20, thickness=5)
    return img

def draw_bbox(img: np.ndarray, bboxes: List[BBox]) -> np.ndarray:
    img = img.copy()
    for (start, end) in bboxes:
        cv2.rectangle(img, start, end, BBOX_COLOR, thickness=2)
    return img

def handle_click(
    img: np.ndarray,
    original_image: np.ndarray,
    selected_points: List[Tuple[Point, int]],
    bounding_boxes: List[BBox],
    temp_point: Optional[Point],
    tool: str,
    point_type: str,
    evt: gr.SelectData
) -> Tuple[np.ndarray, List[Tuple[Point, int]], List[BBox], Optional[Point]]:
    if tool == "Точки":
        label = 1 if point_type == "foreground" else 0
        selected_points.append((evt.index, label))
        return draw_points(img, selected_points), selected_points, bounding_boxes, temp_point

    if tool == "Bounding Box":
        if temp_point is None:
            selected_points.append((evt.index, 2))
            return draw_points(img, selected_points), selected_points, bounding_boxes, evt.index
        if bounding_boxes:
            bounding_boxes.pop()
        bounding_boxes.append((temp_point, evt.index))
        img = undo_point(original_image, selected_points, bounding_boxes)
        return img, selected_points, bounding_boxes, None

    return img, selected_points, bounding_boxes, temp_point

def undo_point(orig_img: np.ndarray, sel_pix: List[Tuple[Point, int]], bboxes: List[BBox]) -> np.ndarray:
    if sel_pix:
        sel_pix.pop()
    img = draw_bbox(orig_img, bboxes)
    return draw_points(img, sel_pix)

def undo_bbox(orig_img: np.ndarray, bboxes: List[BBox], sel_pix: List[Tuple[Point, int]]) -> np.ndarray:
    if bboxes:
        bboxes.pop()
    img = draw_points(orig_img, sel_pix)
    return draw_bbox(img, bboxes)

def clear_all(orig_img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[Point, int]], List[BBox], Optional[Point]]:
    return orig_img, [], [], None

def run_prediction(
    original_image: np.ndarray,
    selected_points: List[Tuple[Point, int]],
    bounding_boxes: List[BBox],
    guidance_mode: str,
    background_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if original_image is None:
        return None, None, None
    formatted_points = [((x, y), label) for ((x, y), label) in selected_points]
    com_img, green_img, alpha_rgb = model_image.predict(
        image=original_image,
        points=formatted_points,
        bboxes=bounding_boxes,
        guidance_mode=guidance_mode,
        background_type=background_type
    )
    return com_img, green_img, alpha_rgb

def update_ui(tool: str):
    return (
        gr.update(visible=(tool == "Точки")),
        gr.update(visible=(tool == "Точки")),
        gr.update(visible=(tool == "Bounding Box"))
    )

# -------------- Функции для работы с видео --------------
# Видео-аннотации: для каждого кадра (ключ) хранится (original_frame, selected_points, bounding_boxes, temp_bbox_start)

def init_video(video_path: str) -> Tuple[dict, dict, np.ndarray, np.ndarray, List[Tuple[Point, int]], List[BBox], Optional[Point], object, int]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {}, {}, None, None, [], [], None, gr.update(), 0
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_info = {"video_path": video_path, "total_frames": total_frames}
    video_annotations = {}
    original_frame = frame.copy()
    selected_points = []
    bounding_boxes = []
    temp_bbox_start = None
    video_annotations[0] = (original_frame, selected_points, bounding_boxes, temp_bbox_start)
    displayed_frame = draw_points(draw_bbox(original_frame, bounding_boxes), selected_points)
    current_frame = 0
    slider_update = gr.update(value=0, maximum=total_frames - 1)
    return video_info, video_annotations, displayed_frame, original_frame, selected_points, bounding_boxes, temp_bbox_start, slider_update, current_frame

def change_frame(new_frame: int, current_frame: int, video_info: dict, video_annotations: dict,
                 original_frame: np.ndarray, selected_points: List[Tuple[Point, int]],
                 bounding_boxes: List[BBox], temp_bbox_start: Optional[Point]
                ) -> Tuple[np.ndarray, dict, np.ndarray, List[Tuple[Point, int]], List[BBox], Optional[Point], int]:
    video_annotations[current_frame] = (original_frame, selected_points, bounding_boxes, temp_bbox_start)
    if new_frame in video_annotations:
        ann = video_annotations[new_frame]
    else:
        cap = cv2.VideoCapture(video_info["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            disp = draw_points(draw_bbox(original_frame, bounding_boxes), selected_points)
            return disp, video_annotations, original_frame, selected_points, bounding_boxes, temp_bbox_start, current_frame
        original_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ann = (original_new, [], [], None)
        video_annotations[new_frame] = ann
    new_original, new_selected_points, new_bounding_boxes, new_temp_bbox_start = ann
    disp = draw_points(draw_bbox(new_original, new_bounding_boxes), new_selected_points)
    return disp, video_annotations, new_original, new_selected_points, new_bounding_boxes, new_temp_bbox_start, new_frame

def read_video_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return np.array(frames)

# Новый метод video_predict (возвращает dict: {frame_idx: (com_img, green_img, alpha_rgb), ...})
def video_predict(video_path: str, all_clicks: dict, all_boxes: dict) -> dict:
    print("Вызван video_predict с параметрами:")
    print("Путь до видео:", video_path)
    print("Все клики:", all_clicks)
    print("Все боксы:", all_boxes)
    
    
    inference_state = model_video_sam2.init_state(video_path=video_path)
    ann_obj_id = 1
    
    
    formatted_annotations = format_annotations_for_predictor(all_clicks, all_boxes)
    
    for frame_idx, (points, labels, box) in formatted_annotations.items():
        model_video_sam2.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
            box=box
        )
    
    for out_frame_idx, out_obj_ids, out_mask_logits in model_video_sam2.propagate_in_video(inference_state):
        continue
    
    image_embed, full_masks, low_masks = inference_state["image_embed"], inference_state["full_masks"], inference_state["low_masks"]
    original_image_np = read_video_frames(video_path)
    
    result = model_image.predict_video(
        image_embed=image_embed,
        full_masks=full_masks, 
        low_masks=low_masks, 
        original_image_np=original_image_np)
    
    
    return result

def format_annotations_for_predictor(all_clicks: dict, all_boxes: dict) -> dict:
    formatted = {}
    for frame, clicks in all_clicks.items():
        points_list = []
        labels_list = []
        for click in clicks:
            pt, lbl = click
            points_list.append(pt)
            labels_list.append(lbl)
        if points_list:
            points_arr = np.array(points_list, dtype=np.float32)
            labels_arr = np.array(labels_list, dtype=np.int32)
        else:
            points_arr = np.empty((0, 2), dtype=np.float32)
            labels_arr = np.empty((0,), dtype=np.int32)
        box = None
        if frame in all_boxes and all_boxes[frame]:
            first_box = all_boxes[frame][0]
            box = np.array([first_box[0][0], first_box[0][1], first_box[1][0], first_box[1][1]], dtype=np.float32)
        formatted[frame] = (points_arr, labels_arr, box)
    return formatted

def collect_annotations(video_annotations: dict) -> Tuple[dict, dict]:
    all_clicks = {}
    all_boxes = {}
    for frame_num, (orig, sel_points, boxes, temp) in video_annotations.items():
        all_clicks[frame_num] = sel_points
        all_boxes[frame_num] = boxes
    return all_clicks, all_boxes

def call_video_predict(video_info: dict, video_annotations: dict) -> str:
    all_clicks, all_boxes = collect_annotations(video_annotations)
    video_predict(video_info.get("video_path", ""), all_clicks, all_boxes)
    return "Video predict called. Check console for parameters."

# Функция для записи видео из массива кадров с использованием метаданных из оригинального файла
def write_video(frames: np.ndarray, video_path: str) -> str:
    # Извлекаем метаданные из оригинального видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Сохраняем временный файл с расширением .mp4
    temp_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    
    # Создаем клип из последовательности кадров (frames уже в формате RGB)
    clip = ImageSequenceClip(list(frames), fps=fps)
    
    # Записываем видео с использованием кодека libx264 (видео будет совместимо с большинством плееров)
    clip.write_videofile(temp_path, codec="libx264", audio=False)
    
    return temp_path

def visualize_video_predictions(video_path: str, predictions: dict) -> np.ndarray:
    frames = read_video_frames(video_path)
    num_frames = frames.shape[0]
    H, W, _ = frames[0].shape
    output_frames = []
    for i in range(num_frames):
        if i in predictions:
            com_img, green_img, alpha_rgb = predictions[i]
            output_frames.append(com_img)
        else:
            output_frames.append(np.zeros((H, W, 3), dtype=np.uint8))
    return np.array(output_frames)

def predict_and_visualize_video(video_info: dict, video_annotations: dict) -> str:
    all_clicks, all_boxes = collect_annotations(video_annotations)
    predictions = video_predict(video_info.get("video_path", ""), all_clicks, all_boxes)
    vis_frames = visualize_video_predictions(video_info.get("video_path", ""), predictions)
    video_file = write_video(vis_frames, video_info.get("video_path", ""))
    return video_file

# -------------- Главная функция с интерфейсом Gradio --------------

def main() -> None:
    with gr.Blocks() as demo:
        gr.Markdown("## Выбор режима работы: Изображения или Видео")
        with gr.Tabs():
            # Вкладка для работы с изображениями
            with gr.TabItem("Изображения"):
                gr.Markdown("### Работа с изображениями")
                input_image = gr.Image(type="numpy", label="Загрузите изображение")
                original_image = gr.State(value=None)
                selected_points = gr.State([])
                bounding_boxes = gr.State([])
                temp_bbox_start = gr.State(value=None)
                with gr.Row():
                    tool_selector = gr.Radio(["Точки", "Bounding Box"], value="Точки", label="Режим работы")
                with gr.Row():
                    radio = gr.Radio(["foreground", "background"], value="foreground", label="Тип точки", visible=True)
                with gr.Row():
                    undo_button = gr.Button("Отменить клик", visible=True)
                    undo_bbox_button = gr.Button("Отменить Bounding Box", visible=False)
                    clear_button = gr.Button("Очистить всё")
                with gr.Row():
                    guidance_mode = gr.Radio(["alpha", "mask"], value="alpha", label="Guidance Mode")
                    background_type = gr.Radio(["real_world_sample", "original"], value="real_world_sample", label="Background Type")
                predict_button = gr.Button("Предсказать")
                with gr.Row():
                    com_img = gr.Image(type="numpy", label="Composited Image")
                    green_img = gr.Image(type="numpy", label="Green Screen")
                    alpha_rgb = gr.Image(type="numpy", label="Alpha Mask")
                input_image.upload(store_img, inputs=[input_image],
                                   outputs=[input_image, original_image, selected_points, bounding_boxes, temp_bbox_start])
                input_image.select(
                    handle_click,
                    inputs=[input_image, original_image, selected_points, bounding_boxes, temp_bbox_start, tool_selector, radio],
                    outputs=[input_image, selected_points, bounding_boxes, temp_bbox_start]
                )
                undo_button.click(undo_point, inputs=[original_image, selected_points, bounding_boxes], outputs=[input_image])
                undo_bbox_button.click(undo_bbox, inputs=[original_image, bounding_boxes, selected_points], outputs=[input_image])
                clear_button.click(clear_all, inputs=[original_image], outputs=[input_image, selected_points, bounding_boxes, temp_bbox_start])
                predict_button.click(
                    run_prediction,
                    inputs=[original_image, selected_points, bounding_boxes, guidance_mode, background_type],
                    outputs=[com_img, green_img, alpha_rgb]
                )
                tool_selector.change(update_ui, inputs=[tool_selector], outputs=[radio, undo_button, undo_bbox_button])
            
            # Вкладка для работы с видео
            with gr.TabItem("Видео"):
                gr.Markdown("### Работа с видео")
                video_input = gr.Video(label="Загрузите видео")
                frame_selector = gr.Slider(0, 1, step=1, label="Выберите кадр")
                video_frame = gr.Image(type="numpy", label="Кадр для аннотации")
                video_info = gr.State(value={})
                video_annotations = gr.State(value={})
                current_frame = gr.State(value=0)
                original_frame = gr.State(value=None)
                selected_points_video = gr.State(value=[])
                bounding_boxes_video = gr.State(value=[])
                temp_bbox_start_video = gr.State(value=None)
                with gr.Row():
                    tool_selector_video = gr.Radio(["Точки", "Bounding Box"], value="Точки", label="Режим работы")
                with gr.Row():
                    radio_video = gr.Radio(["foreground", "background"], value="foreground", label="Тип точки", visible=True)
                with gr.Row():
                    undo_button_video = gr.Button("Отменить клик", visible=True)
                    undo_bbox_button_video = gr.Button("Отменить Bounding Box", visible=False)
                    clear_button_video = gr.Button("Очистить всё")
                with gr.Row():
                    guidance_mode_video = gr.Radio(["alpha", "mask"], value="alpha", label="Guidance Mode")
                    background_type_video = gr.Radio(["real_world_sample", "original"], value="real_world_sample", label="Background Type")
                predict_button_video = gr.Button("Предсказать")
                video_predict_button = gr.Button("Предсказать видео")
                visualize_predictions_button = gr.Button("Визуализировать предсказания")
                video_output = gr.Video(label="Видео с предсказаниями")
                with gr.Row():
                    com_img_video = gr.Image(type="numpy", label="Composited Image")
                    green_img_video = gr.Image(type="numpy", label="Green Screen")
                    alpha_rgb_video = gr.Image(type="numpy", label="Alpha Mask")
                
                video_input.change(
                    init_video,
                    inputs=[video_input],
                    outputs=[video_info, video_annotations, video_frame, original_frame,
                             selected_points_video, bounding_boxes_video, temp_bbox_start_video,
                             frame_selector, current_frame]
                )
                frame_selector.change(
                    change_frame,
                    inputs=[frame_selector, current_frame, video_info, video_annotations,
                            original_frame, selected_points_video, bounding_boxes_video, temp_bbox_start_video],
                    outputs=[video_frame, video_annotations, original_frame, selected_points_video, bounding_boxes_video, temp_bbox_start_video, current_frame]
                )
                video_frame.select(
                    handle_click,
                    inputs=[video_frame, original_frame, selected_points_video, bounding_boxes_video, temp_bbox_start_video, tool_selector_video, radio_video],
                    outputs=[video_frame, selected_points_video, bounding_boxes_video, temp_bbox_start_video]
                )
                undo_button_video.click(undo_point, inputs=[original_frame, selected_points_video, bounding_boxes_video], outputs=[video_frame])
                undo_bbox_button_video.click(undo_bbox, inputs=[original_frame, bounding_boxes_video, selected_points_video], outputs=[video_frame])
                clear_button_video.click(clear_all, inputs=[original_frame], outputs=[video_frame, selected_points_video, bounding_boxes_video, temp_bbox_start_video])
                predict_button_video.click(
                    run_prediction,
                    inputs=[original_frame, selected_points_video, bounding_boxes_video, guidance_mode_video, background_type_video],
                    outputs=[com_img_video, green_img_video, alpha_rgb_video]
                )
                tool_selector_video.change(update_ui, inputs=[tool_selector_video], outputs=[radio_video, undo_button_video, undo_bbox_button_video])
                video_predict_button.click(
                    call_video_predict,
                    inputs=[video_info, video_annotations],
                    outputs=[]
                )
                visualize_predictions_button.click(
                    predict_and_visualize_video,
                    inputs=[video_info, video_annotations],
                    outputs=[video_output]
                )
        demo.launch(debug=True)

if __name__ == "__main__":
    main()
