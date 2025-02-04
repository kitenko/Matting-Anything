import av
import cv2
import numpy as np
import gradio as gr
import tempfile
from typing import List, Tuple, Optional

import fractions


# Import your predictor/inference class
from inference import MAMInferencer
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor_hf

# Build the video predictor instance using a SAM2 model from HuggingFace.
model_video_sam2: SAM2VideoPredictor = build_sam2_video_predictor_hf("facebook/sam2.1-hiera-base-plus")

# Define types for clarity
Point = Tuple[int, int]   # Coordinates of a point (x, y)
BBox = Tuple[Point, Point] # Bounding Box: defined by top-left and bottom-right corners

# Define colors and marker types for drawing annotations on images.
COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # Example: Red (foreground), Blue (background), Green (for bounding box points)
MARKERS = [1, 5, 3]  # Marker types used by OpenCV's drawMarker function
BBOX_COLOR = (0, 255, 0)  # Green color for drawing bounding boxes

# -------------- Image Utility Functions --------------

def store_img(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[Point, int]], List[BBox], Optional[Point]]:
    """
    Store the uploaded image and initialize its associated annotation states.
    
    Returns:
        - The original image.
        - A copy of the image (to be used for drawing annotations).
        - An empty list for selected points.
        - An empty list for bounding boxes.
        - A placeholder (None) for a temporary bounding box starting point.
    """
    return img, img.copy(), [], [], None

def draw_points(img: np.ndarray, points: List[Tuple[Point, int]]) -> np.ndarray:
    """
    Draw markers at each specified point on a copy of the image.

    Args:
        img: The image on which to draw.
        points: List of tuples, each containing a point (x, y) and a label (which determines the color and marker).

    Returns:
        The image with drawn points.
    """
    img = img.copy()
    for point, label in points:
        # Use the marker type and color based on the label
        cv2.drawMarker(img, point, COLORS[label], markerType=MARKERS[label], markerSize=20, thickness=5)
    return img

def draw_bbox(img: np.ndarray, bboxes: List[BBox]) -> np.ndarray:
    """
    Draw bounding boxes on a copy of the image.

    Args:
        img: The image on which to draw.
        bboxes: List of bounding boxes, each defined by two points.

    Returns:
        The image with drawn bounding boxes.
    """
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
    """
    Process a user click on the image, updating the annotation state based on the selected tool.
    
    Args:
        img: The current annotated image.
        original_image: The original image without annotations.
        selected_points: List of currently selected points.
        bounding_boxes: List of current bounding boxes.
        temp_point: Temporary starting point for bounding box drawing.
        tool: The current tool ("Точки" for points or "Bounding Box" for box annotations).
        point_type: The type of point (e.g., "foreground" or "background").
        evt: The event data from Gradio that includes the click index (coordinates).

    Returns:
        A tuple containing the updated annotated image, updated selected points,
        updated bounding boxes, and possibly updated temporary bounding box start point.
    """
    # If the tool is (Points)
    if tool == "Points":
        # Label 1 for foreground, label 0 for background
        label = 1 if point_type == "foreground" else 0
        selected_points.append((evt.index, label))
        # Redraw the image with the new point
        return draw_points(img, selected_points), selected_points, bounding_boxes, temp_point

    # If the tool is "Bounding Box"
    if tool == "Bounding Box":
        # If no starting point is set, use the click as the starting point and add a temporary point
        if temp_point is None:
            selected_points.append((evt.index, 2))  # label 2 used to mark the bbox start point
            return draw_points(img, selected_points), selected_points, bounding_boxes, evt.index
        # Otherwise, complete the bounding box by pairing the starting point with the new click point
        if bounding_boxes:
            bounding_boxes.pop()  # Remove previous bbox if any (only one bbox is allowed at a time)
        bounding_boxes.append((temp_point, evt.index))
        # Redraw the image from the original image (to avoid cumulative drawing errors)
        img = undo_point(original_image, selected_points, bounding_boxes)
        return img, selected_points, bounding_boxes, None

    # If none of the tools match, return the unchanged image and state.
    return img, selected_points, bounding_boxes, temp_point

def undo_point(orig_img: np.ndarray, sel_pix: List[Tuple[Point, int]], bboxes: List[BBox]) -> np.ndarray:
    """
    Undo the last point annotation.

    Args:
        orig_img: The original image.
        sel_pix: The list of current selected points.
        bboxes: The list of current bounding boxes.

    Returns:
        The image after removing the last point (redrawing both boxes and points).
    """
    if sel_pix:
        sel_pix.pop()  # Remove the last point
    img = draw_bbox(orig_img, bboxes)
    return draw_points(img, sel_pix)

def undo_bbox(orig_img: np.ndarray, bboxes: List[BBox], sel_pix: List[Tuple[Point, int]]) -> np.ndarray:
    """
    Undo the last bounding box annotation.

    Args:
        orig_img: The original image.
        bboxes: The list of bounding boxes.
        sel_pix: The list of selected points.

    Returns:
        The image after removing the last bounding box.
    """
    if bboxes:
        bboxes.pop()  # Remove the last bounding box
    img = draw_points(orig_img, sel_pix)
    return draw_bbox(img, bboxes)

def clear_all(orig_img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[Point, int]], List[BBox], Optional[Point]]:
    """
    Clear all annotations, returning the original image and empty annotation states.
    
    Args:
        orig_img: The original image.

    Returns:
        A tuple containing the original image, an empty list of points, an empty list of bounding boxes, and None.
    """
    return orig_img, [], [], None

def clear_all_annotation_video(orig_img: np.ndarray):
    return orig_img, [], [], {}

def run_prediction(
    original_image: np.ndarray,
    selected_points: List[Tuple[Point, int]],
    bounding_boxes: List[BBox],
    guidance_mode: str,
    background_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the prediction using the image inference model.

    Args:
        original_image: The original image to predict on.
        selected_points: Annotated points formatted as (x, y, label).
        bounding_boxes: Annotated bounding boxes.
        guidance_mode: A mode that controls how guidance is applied (e.g., "alpha" or "mask").
        background_type: Specifies which background to use (e.g., "real_world_sample" or "original").

    Returns:
        A tuple of three images: composited image, green screen image, and alpha mask.
    """
    if original_image is None:
        return None, None, None
    # Format the points for the model; note that the formatting here is trivial.
    formatted_points = [((x, y), label) for ((x, y), label) in selected_points]
    com_img, green_img, alpha_rgb = model_image.predict(
        image=original_image,
        points=formatted_points,
        bboxes=bounding_boxes,
        guidance_mode=guidance_mode,
        background_type=background_type
    )
    return com_img, green_img, alpha_rgb, selected_points, bounding_boxes

def update_ui(tool: str):
    """
    Update the UI elements based on the selected tool.
    Shows or hides certain buttons and controls.

    Args:
        tool: The selected annotation tool.

    Returns:
        A tuple of updated UI components.
    """
    return (
        gr.update(visible=(tool == "Points")),
        gr.update(visible=(tool == "Points")),
        gr.update(visible=(tool == "Bounding Box"))
    )

# -------------- Video Utility Functions --------------
# Video annotations: For each frame (key), store a tuple:
# (original_frame, selected_points, bounding_boxes, temp_bbox_start)

def init_video(video_path: str) -> Tuple[dict, dict, np.ndarray, np.ndarray, List[Tuple[Point, int]], List[BBox], Optional[Point], object, int]:
    """
    Initialize the video by reading the first frame and setting up annotation states.

    Args:
        video_path: The path to the video file.

    Returns:
        A tuple containing:
            - video_info: A dictionary with video path and total frame count.
            - video_annotations: A dictionary mapping frame indices to their annotations.
            - displayed_frame: The frame with initial annotations drawn.
            - original_frame: The raw first frame (RGB).
            - selected_points: An empty list for points.
            - bounding_boxes: An empty list for bounding boxes.
            - temp_bbox_start: None (no temporary bounding box starting point yet).
            - slider_update: A Gradio update object for the frame slider.
            - current_frame: The current frame index (0 initially).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {}, {}, None, None, [], [], None, gr.update(), 0
    # Convert the first frame from BGR to RGB.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_info = {"video_path": video_path, "total_frames": total_frames}
    video_annotations = {}
    original_frame = frame.copy()
    selected_points = []
    bounding_boxes = []
    temp_bbox_start = None
    # Store annotation data for frame 0
    video_annotations[0] = (original_frame, selected_points, bounding_boxes, temp_bbox_start)
    # Draw any existing annotations (currently none) on the first frame.
    displayed_frame = draw_points(draw_bbox(original_frame, bounding_boxes), selected_points)
    current_frame = 0
    slider_update = gr.update(value=0, maximum=total_frames - 1)
    return video_info, video_annotations, displayed_frame, original_frame, selected_points, bounding_boxes, temp_bbox_start, slider_update, current_frame

def change_frame(new_frame: int, current_frame: int, video_info: dict, video_annotations: dict,
                 original_frame: np.ndarray, selected_points: List[Tuple[Point, int]],
                 bounding_boxes: List[BBox], temp_bbox_start: Optional[Point]
                ) -> Tuple[np.ndarray, dict, np.ndarray, List[Tuple[Point, int]], List[BBox], Optional[Point], int]:
    """
    Change the current frame in the video annotation interface.
    
    Args:
        new_frame: The new frame index to switch to.
        current_frame: The current frame index.
        video_info: Video metadata.
        video_annotations: Annotation states for frames.
        original_frame: The original frame for the current frame.
        selected_points: Points annotated on the current frame.
        bounding_boxes: Bounding boxes annotated on the current frame.
        temp_bbox_start: Temporary bounding box start point for the current frame.

    Returns:
        A tuple with:
            - The displayed frame (with annotations) for the new frame.
            - Updated video_annotations.
            - The original image for the new frame.
            - Selected points for the new frame.
            - Bounding boxes for the new frame.
            - Temporary bounding box start point for the new frame.
            - The new frame index.
    """
    # Save current frame's annotations
    video_annotations[current_frame] = (original_frame, selected_points, bounding_boxes, temp_bbox_start)
    
    # If the new frame has been annotated before, retrieve its annotations
    if new_frame in video_annotations:
        ann = video_annotations[new_frame]
    else:
        # Otherwise, read the new frame from the video
        cap = cv2.VideoCapture(video_info["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            # If failed to read, return current state's drawing.
            disp = draw_points(draw_bbox(original_frame, bounding_boxes), selected_points)
            return disp, video_annotations, original_frame, selected_points, bounding_boxes, temp_bbox_start, current_frame
        # Convert the frame to RGB and initialize annotations for this frame.
        original_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ann = (original_new, [], [], None)
        video_annotations[new_frame] = ann
    new_original, new_selected_points, new_bounding_boxes, new_temp_bbox_start = ann
    disp = draw_points(draw_bbox(new_original, new_bounding_boxes), new_selected_points)
    return disp, video_annotations, new_original, new_selected_points, new_bounding_boxes, new_temp_bbox_start, new_frame

def read_video_frames(video_path: str) -> np.ndarray:
    """
    Read all frames from a video file and convert them to RGB.

    Args:
        video_path: The path to the video file.

    Returns:
        A NumPy array containing all video frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return np.array(frames)

def video_predict(video_path: str, all_clicks: dict, all_boxes: dict) -> dict:
    """
    Predict masks on a video based on the provided annotations (points and boxes).

    Args:
        video_path: The path to the video.
        all_clicks: Dictionary mapping frame indices to lists of point annotations.
        all_boxes: Dictionary mapping frame indices to lists of bounding boxes.

    Returns:
        A dictionary mapping frame indices to prediction results.
    """
    print("Called video_predict with parameters:")
    print("Video path:", video_path)
    print("All clicks:", all_clicks)
    print("All boxes:", all_boxes)
    
    # Initialize inference state for the video (used for propagating annotations)
    inference_state = model_video_sam2.init_state(video_path=video_path)
    ann_obj_id = 1  # Annotation object ID (could be used to track multiple objects)
    
    # Format annotations for the predictor
    formatted_annotations = format_annotations_for_predictor(all_clicks, all_boxes)
    
    # For each frame's annotations, add the points/boxes to the inference state
    for frame_idx, (points, labels, box) in formatted_annotations.items():
        model_video_sam2.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
            box=box
        )
    
    # Propagate annotations through the video; this loop runs until propagation is complete.
    for _ in model_video_sam2.propagate_in_video(inference_state):
        continue
    
    # Extract the required outputs for predictions
    image_embed = inference_state["image_embed"]
    full_masks = inference_state["full_masks"]
    low_masks = inference_state["low_masks"]
    original_image_np = read_video_frames(video_path)
    
    # Get the final video prediction from the image inference model.
    result = model_image.predict_video(
        image_embed=image_embed,
        full_masks=full_masks, 
        low_masks=low_masks, 
        original_image_np=original_image_np
    )
    
    return result

def format_annotations_for_predictor(all_clicks: dict, all_boxes: dict) -> dict:
    """
    Format the raw click and bounding box annotations for the predictor.

    Args:
        all_clicks: A dictionary mapping frame indices to point annotations.
        all_boxes: A dictionary mapping frame indices to bounding box annotations.

    Returns:
        A dictionary mapping frame indices to a tuple:
        (points array, labels array, bounding box array or None)
    """
    formatted = {}
    for frame, clicks in all_clicks.items():
        # Get bounding boxes for this frame; if none, use an empty list.
        boxes = all_boxes.get(frame, [])
        if not clicks and not boxes:
            continue

        # Format points and labels into numpy arrays if points exist.
        points_arr = (
            np.array([pt for pt, _ in clicks], dtype=np.float32)
            if clicks else None
        )
        labels_arr = (
            np.array([lbl for _, lbl in clicks], dtype=np.int32)
            if clicks else None
        )

        # If there is at least one bounding box, convert the first one to an array.
        box = None
        if boxes:
            (x0, y0), (x1, y1) = boxes[0]
            box = np.array([x0, y0, x1, y1], dtype=np.float32)

        formatted[frame] = (points_arr, labels_arr, box)
    return formatted

def collect_annotations(video_annotations: dict) -> Tuple[dict, dict]:
    """
    Collect all click and bounding box annotations from all video frames.

    Args:
        video_annotations: Dictionary mapping frame indices to their annotations.

    Returns:
        A tuple of two dictionaries:
            - all_clicks: Frame-wise point annotations.
            - all_boxes: Frame-wise bounding box annotations.
    """
    all_clicks = {}
    all_boxes = {}
    for frame_num, (orig, sel_points, boxes, temp) in video_annotations.items():
        all_clicks[frame_num] = sel_points
        all_boxes[frame_num] = boxes
    return all_clicks, all_boxes

def call_video_predict(video_info: dict, video_annotations: dict) -> str:
    """
    A helper function to trigger video prediction.

    Args:
        video_info: Dictionary containing video metadata.
        video_annotations: Dictionary containing frame-wise annotations.

    Returns:
        A string indicating that video prediction has been called.
    """
    all_clicks, all_boxes = collect_annotations(video_annotations)
    video_predict(video_info.get("video_path", ""), all_clicks, all_boxes)
    return "Video predict called. Check console for parameters."

def write_video_av(frames: np.ndarray, video_path: str) -> str:
    """
    Writes a video from an array of frames using PyAV.

    Arguments:
        frames: An array of frames (in RGB format).
        video_path: The path to the source video (to extract metadata such as FPS).

    Returns:
        The path to the temporarily saved video file.
    """
    # Extract FPS from the source video.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Determine frame dimensions.
    height, width = frames[0].shape[:2]

    # Create a temporary file with the .mp4 extension.
    temp_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    # Open a container for writing.
    container = av.open(temp_path, mode='w')

    # Add a video stream using the libx264 codec.
    rate = fractions.Fraction(fps).limit_denominator()
    stream = container.add_stream('libx264', rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'  # Required format for most players

    # Encoding and adding frames.
    for frame in frames:
        # Create a VideoFrame object from the numpy array (note that the frame format is 'rgb24').
        video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        # When encoding, PyAV converts the frame to the required format (e.g., yuv420p).
        for packet in stream.encode(video_frame):
            container.mux(packet)

    # Finalize encoding: flush buffers.
    for packet in stream.encode():
        container.mux(packet)

    # Close the container.
    container.close()
    return temp_path


def visualize_video_predictions(video_path: str, predictions: dict) -> np.ndarray:
    """
    Visualize predictions on each frame of the video.
    
    Args:
        video_path: The path to the original video.
        predictions: A dictionary mapping frame indices to prediction results.

    Returns:
        An array of frames where each frame is either the predicted image or a black frame.
    """
    frames = read_video_frames(video_path)
    num_frames = frames.shape[0]
    H, W, _ = frames[0].shape
    output_frames = []
    for i in range(num_frames):
        if i in predictions:
            com_img = predictions[i]
            output_frames.append(com_img)
        else:
            # Create a black frame if there is no prediction for this frame.
            output_frames.append(np.zeros((H, W, 3), dtype=np.uint8))
    return np.array(output_frames)

def predict_and_visualize_video(video_info: dict, video_annotations: dict) -> str:
    """
    Run video prediction and generate a video file with visualized predictions.

    Args:
        video_info: Dictionary with video metadata.
        video_annotations: Dictionary with frame-wise annotations.

    Returns:
        Path to the generated video file.
    """
    all_clicks, all_boxes = collect_annotations(video_annotations)
    predictions = video_predict(video_info.get("video_path", ""), all_clicks, all_boxes)
    vis_frames = visualize_video_predictions(video_info.get("video_path", ""), predictions)
    video_file = write_video_av(vis_frames, video_info.get("video_path", ""))
    return video_file

# -------------- Main Gradio Interface --------------

def main() -> None:
    """
    Build and launch the Gradio interface for both image and video annotation/prediction.
    This function sets up two tabs: one for images and one for videos.
    """
    with gr.Blocks() as demo:
        gr.Markdown("## Select Mode: Images or Video")
        with gr.Tabs():
            # Tab for working with images.
            with gr.TabItem("Images"):
                gr.Markdown("### Image Annotation and Prediction")
                input_image = gr.Image(type="numpy", label="Upload an Image")
                original_image = gr.State(value=None)
                selected_points = gr.State([])
                bounding_boxes = gr.State([])
                temp_bbox_start = gr.State(value=None)
                with gr.Row():
                    tool_selector = gr.Radio(["Points", "Bounding Box"], value="Points", label="Annotation Mode")
                with gr.Row():
                    radio = gr.Radio(["foreground", "background"], value="foreground", label="Point Type", visible=True)
                with gr.Row():
                    undo_button = gr.Button("Undo Point", visible=True)
                    undo_bbox_button = gr.Button("Undo Bounding Box", visible=False)
                    clear_button = gr.Button("Clear All Annotation")
                with gr.Row():
                    guidance_mode = gr.Radio(["alpha", "mask"], value="alpha", label="Guidance Mode")
                    background_type = gr.Radio(["real_world_sample", "original"], value="real_world_sample", label="Background Type")
                predict_button = gr.Button("Predict")
                with gr.Row():
                    com_img = gr.Image(type="numpy", label="Composited Image")
                    green_img = gr.Image(type="numpy", label="Green Screen")
                    alpha_rgb = gr.Image(type="numpy", label="Alpha Mask")
                # When an image is uploaded, initialize the annotation states.
                input_image.upload(
                    store_img, 
                    inputs=[input_image],
                    outputs=[input_image, original_image, selected_points, bounding_boxes, temp_bbox_start]
                )
                # Handle clicks on the image for annotations.
                input_image.select(
                    handle_click,
                    inputs=[input_image, original_image, selected_points, bounding_boxes, temp_bbox_start, tool_selector, radio],
                    outputs=[input_image, selected_points, bounding_boxes, temp_bbox_start]
                )
                # Buttons for undoing annotations.
                undo_button.click(
                    undo_point, 
                    inputs=[original_image, selected_points, bounding_boxes], 
                    outputs=[input_image]
                )
                undo_bbox_button.click(
                    undo_bbox, 
                    inputs=[original_image, bounding_boxes, selected_points], 
                    outputs=[input_image]
                )
                clear_button.click(
                    clear_all, 
                    inputs=[original_image], 
                    outputs=[input_image, selected_points, bounding_boxes, temp_bbox_start]
                )
                # Prediction button to run inference on the annotated image.
                predict_button.click(
                    run_prediction,
                    inputs=[original_image, selected_points, bounding_boxes, guidance_mode, background_type],
                    outputs=[com_img, green_img, alpha_rgb]
                )
                # Update the UI when the annotation tool is changed.
                tool_selector.change(
                    update_ui, 
                    inputs=[tool_selector], 
                    outputs=[radio, undo_button, undo_bbox_button]
                )
            
            # Tab for working with videos.
            with gr.TabItem("Video"):
                gr.Markdown("### Video Annotation and Prediction")
                video_input = gr.Video(label="Upload a Video", width="auto", height="auto")
                frame_selector = gr.Slider(0, 1, step=1, label="Select Frame")
                video_frame = gr.Image(type="numpy", label="Frame for Annotation")
                video_info = gr.State(value={})
                video_annotations = gr.State(value={})
                current_frame = gr.State(value=0)
                original_frame = gr.State(value=None)
                selected_points_video = gr.State(value=[])
                bounding_boxes_video = gr.State(value=[])
                temp_bbox_start_video = gr.State(value=None)
                with gr.Row():
                    tool_selector_video = gr.Radio(["Points", "Bounding Box"], value="Points", label="Annotation Mode")
                with gr.Row():
                    radio_video = gr.Radio(["foreground", "background"], value="foreground", label="Point Type", visible=True)
                with gr.Row():
                    undo_button_video = gr.Button("Undo Point", visible=True)
                    undo_bbox_button_video = gr.Button("Undo Bounding Box", visible=False)
                    clear_button_video = gr.Button("Clear All Frame Annotation")
                    clear_button_video_all = gr.Button("Clear All Video Annotation")
                with gr.Row():
                    guidance_mode_video = gr.Radio(["alpha", "mask"], value="alpha", label="Guidance Mode")
                    background_type_video = gr.Radio(["real_world_sample", "original"], value="real_world_sample", label="Background Type")
                predict_button_video = gr.Button("Predict for Current Frame")
                
                with gr.Row():
                    com_img_video = gr.Image(type="numpy", label="Composited Image")
                    green_img_video = gr.Image(type="numpy", label="Green Screen")
                    alpha_rgb_video = gr.Image(type="numpy", label="Alpha Mask")
                    
                visualize_predictions_button = gr.Button("Visualize Predictions for Entire Video")
                video_output = gr.Video(label="Video with Predictions", width="auto", height="auto")
                
                # When a video is uploaded, initialize video annotation states.
                video_input.change(
                    init_video,
                    inputs=[video_input],
                    outputs=[video_info, video_annotations, video_frame, original_frame,
                             selected_points_video, bounding_boxes_video, temp_bbox_start_video,
                             frame_selector, current_frame]
                )
                # When the frame slider changes, update the displayed frame and its annotations.
                frame_selector.change(
                    change_frame,
                    inputs=[frame_selector, current_frame, video_info, video_annotations,
                            original_frame, selected_points_video, bounding_boxes_video, temp_bbox_start_video],
                    outputs=[video_frame, video_annotations, original_frame, selected_points_video, bounding_boxes_video, temp_bbox_start_video, current_frame]
                )
                # Handle clicks on the video frame for annotations.
                video_frame.select(
                    handle_click,
                    inputs=[video_frame, original_frame, selected_points_video, bounding_boxes_video, temp_bbox_start_video, tool_selector_video, radio_video],
                    outputs=[video_frame, selected_points_video, bounding_boxes_video, temp_bbox_start_video]
                )
                # Undo and clear functions for video frame annotations.
                undo_button_video.click(
                    undo_point, 
                    inputs=[original_frame, selected_points_video, bounding_boxes_video], 
                    outputs=[video_frame]
                )
                undo_bbox_button_video.click(
                    undo_bbox, 
                    inputs=[original_frame, bounding_boxes_video, selected_points_video], 
                    outputs=[video_frame]
                )
                clear_button_video.click(
                    clear_all, 
                    inputs=[original_frame], 
                    outputs=[video_frame, selected_points_video, bounding_boxes_video, temp_bbox_start_video]
                )
                
                clear_button_video_all.click(
                    clear_all_annotation_video,
                    inputs=[original_frame],  
                    outputs=[video_frame, selected_points_video, bounding_boxes_video, video_annotations]
                )
                
                # Predict for the current frame using the image predictor.
                predict_button_video.click(
                    run_prediction,
                    inputs=[original_frame, selected_points_video, bounding_boxes_video, guidance_mode_video, background_type_video],
                    outputs=[com_img_video, green_img_video, alpha_rgb_video, selected_points_video, bounding_boxes_video]
                )
                # Update the UI for video annotations when tool changes.
                tool_selector_video.change(
                    update_ui, 
                    inputs=[tool_selector_video], 
                    outputs=[radio_video, undo_button_video, undo_bbox_button_video]
                )
                
                # When the "Visualize Predictions" button is clicked,
                # run video prediction and return a video file with visualized predictions.
                visualize_predictions_button.click(
                    predict_and_visualize_video,
                    inputs=[video_info, video_annotations],
                    outputs=[video_output]
                )
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
        

# Create an image inference model that uses the video SAM2 predictor.
# Note: This object is shared with the video predictor model.
model_image = None

def init_models(model_weights_path: str) -> None:
    global model_image
    model_image = MAMInferencer(model_video_sam2, model_weights_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image and Video Alpha Channel Prediction")
    parser.add_argument(
        "--model-weights",
        type=str,
        required=True,
        help="Path to the model weights file (e.g., /app/checkpoints/your_model.pth)"
    )
    args = parser.parse_args()

    init_models(args.model_weights)

    main()
