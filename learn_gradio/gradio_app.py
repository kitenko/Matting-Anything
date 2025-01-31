import cv2
import numpy as np
import gradio as gr
from typing import List, Tuple, Optional

# Импортируем ваш класс с методом `predict`
import sys
sys.path.append("/app")
from learn_gradio.inference import MAMInferencer

# Создаём экземпляр модели
model = MAMInferencer("/app/checkpoints/return_cousine_pre_train_grad_true_new_shedule_real_world_aug_full_data_sam_2_multiple_mask_True/model_step_30000.pth") 

# Определяем типы данных
Point = Tuple[int, int]  # Координаты точки (x, y)
BBox = Tuple[Point, Point]  # Bounding Box (верхний левый и нижний правый угол)

# Цвета и маркеры
COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # Красный (foreground), Синий (background)
MARKERS = [1, 5, 3]
BBOX_COLOR = (0, 255, 0)  # Bounding Box - зелёный

# 🔹 Обработчик загрузки изображения
def store_img(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[Point, int]], List[BBox], Optional[Point]]:
    """Сохраняет оригинальное изображение и очищает списки точек и BBox."""
    return img, img.copy(), [], [], None  # Сохраняем original_image, очищаем точки и BBox

def draw_points(img: np.ndarray, points: List[Tuple[Point, int]]) -> np.ndarray:
    """Рисует точки на изображении."""
    img = img.copy()
    for point, label in points:
        cv2.drawMarker(img, point, COLORS[label], markerType=MARKERS[label], markerSize=20, thickness=5)
    return img

def draw_bbox(img: np.ndarray, bboxes: List[BBox]) -> np.ndarray:
    """Рисует все Bounding Box на изображении."""
    img = img.copy()
    for (start, end) in bboxes:
        cv2.rectangle(img, start, end, BBOX_COLOR, thickness=2)
    return img


# 🔹 Обработчик кликов (работает для точек и Bounding Box)
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
    """Обрабатывает клик: добавляет точку или создаёт Bounding Box."""
    
    if tool == "Точки":
        label = 0 if point_type == "foreground" else 1
        selected_points.append((evt.index, label))  # Добавляем точку
        return draw_points(img, selected_points), selected_points, bounding_boxes, temp_point

    if tool == "Bounding Box":
        if temp_point is None:
            selected_points.append((evt.index, 2))  # Временная точка для BBox
            return draw_points(img, selected_points), selected_points, bounding_boxes, evt.index
        
        if bounding_boxes:
            bounding_boxes.pop()
        # Завершаем Bounding Box
        bounding_boxes.append((temp_point, evt.index))
        img = undo_point(original_image, selected_points, bounding_boxes)  # Очищаем временные точки
        return img, selected_points, bounding_boxes, None

    return img, selected_points, bounding_boxes, temp_point  # На случай ошибки

# 🔹 Отмена последней точки
def undo_point(orig_img: np.ndarray, sel_pix: List[Tuple[Point, int]], bboxes: List[BBox]) -> np.ndarray:
    """Отменяет последний клик и обновляет изображение."""
    if sel_pix:
        sel_pix.pop()  # Удаляем последнюю точку
    img = draw_bbox(orig_img, bboxes)
    return draw_points(img, sel_pix)

# 🔹 Отмена последнего Bounding Box
def undo_bbox(orig_img: np.ndarray, bboxes: List[BBox], sel_pix: List[Tuple[Point, int]]) -> np.ndarray:
    """Отменяет последний Bounding Box и обновляет изображение."""
    if bboxes:
        bboxes.pop()  # Удаляем последний BBox
    img = draw_points(orig_img, sel_pix)
    return draw_bbox(img, bboxes)

# 🔹 Очистка всего
def clear_all(orig_img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[Point, int]], List[BBox], Optional[Point]]:
    """Очищает все точки и Bounding Box."""
    return orig_img, [], [], None

# 🔹 Запуск предсказания
def run_prediction(
    original_image: np.ndarray,
    selected_points: List[Tuple[Point, int]],
    bounding_boxes: List[BBox],
    guidance_mode: str,
    background_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Выполняет инференс модели и возвращает три изображения."""
    
    if original_image is None:
        return None, None, None
    
    formatted_points = [((x, y), label) for ((x, y), label) in selected_points]
    
    com_img, green_img, alpha_rgb = model.predict(
        image=original_image,
        points=formatted_points,
        bboxes=bounding_boxes,
        guidance_mode=guidance_mode,
        background_type=background_type
    )

    return com_img, green_img, alpha_rgb

# 🔹 Функция обновления UI
def update_ui(tool: str):
    """Обновляет UI в зависимости от выбранного режима работы."""
    return (
        gr.update(visible=(tool == "Точки")),
        gr.update(visible=(tool == "Точки")),
        gr.update(visible=(tool == "Bounding Box"))
    )

# 🔹 Главная функция для запуска Gradio
def main() -> None:
    with gr.Blocks() as demo:
        gr.Markdown("### Выбор точек, Bounding Box и запуск инференса")

        input_image = gr.Image(type="numpy")
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

        input_image.upload(store_img, [input_image], [input_image, original_image, selected_points, bounding_boxes, temp_bbox_start])

        input_image.select(
            handle_click, 
            [input_image, original_image, selected_points, bounding_boxes, temp_bbox_start, tool_selector, radio],  
            [input_image, selected_points, bounding_boxes, temp_bbox_start]
        )

        undo_button.click(undo_point, [original_image, selected_points, bounding_boxes], [input_image])
        undo_bbox_button.click(undo_bbox, [original_image, bounding_boxes, selected_points], [input_image])
        clear_button.click(clear_all, [original_image], [input_image, selected_points, bounding_boxes, temp_bbox_start])

        predict_button.click(
            run_prediction,
            [original_image, selected_points, bounding_boxes, guidance_mode, background_type],
            [com_img, green_img, alpha_rgb]
        )

        tool_selector.change(update_ui, [tool_selector], [radio, undo_button, undo_bbox_button])

    demo.launch(debug=True)

if __name__ == "__main__":
    main()
