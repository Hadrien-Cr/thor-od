from pathlib import Path
import ai2thor.platform
from ai2thor.controller import Controller
from utils.scene import get_scene_type


def setup_detector(weights_path: Path, device: str):
    from data_collection.detector import YOLOv5Detector
    detector = YOLOv5Detector(
        model_path=weights_path,
        yolo_project_path=constants.YOLOV5_PROJECT_PATH,
        device=device,
    )
    return detector


def setup_controller(
    scene_name: str,
    grid_size: float,
    visibility_distance: float,
    rotate_step_degrees: int,
    render_instance_segmentation: bool,
    render_depth_image: bool,
    snap_to_grid: bool,
    continuous: bool,
    cloud_rendering: bool,
    id: int = 0,
    quality: str = "Ultra",
):
    scene_type = constants.get_scene_type(scene_name)
    if scene_type == "procthor":
        scene = constants.scene_name_to_scene_spec(scene_name)
    else:
        scene = scene_name

    if cloud_rendering:
        controller = Controller(
            platform=ai2thor.platform.CloudRendering,
            scene=scene,
            visibilityDistance=visibility_distance,
            gridSize=grid_size,
            height=constants.IMG_SIZE,
            width=constants.IMG_SIZE,
            rotateStepDegrees=rotate_step_degrees,
            renderInstanceSegmentation=render_instance_segmentation,
            renderDepthImage=render_depth_image,
            snapToGrid=snap_to_grid,
            continuous=continuous,
            host=f"127.0.0.{id}",
            quality=quality,
        )
        controller.scene_name = scene_name
        return controller
    else:
        controller = Controller(
            scene=scene,
            gridSize=grid_size,
            height=constants.IMG_SIZE,
            width=constants.IMG_SIZE,
            visibilityDistance=visibility_distance,
            rotateStepDegrees=rotate_step_degrees,
            renderInstanceSegmentation=render_instance_segmentation,
            renderDepthImage=render_depth_image,
            snapToGrid=snap_to_grid,
            continuous=continuous,
            host=f"127.0.0.{id}",
            quality=quality,
        )
        controller.scene_name = scene_name
        return controller
