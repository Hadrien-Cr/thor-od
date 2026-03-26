import itertools 
import numpy as np
from dataclasses import dataclass
from ai2thor.server import Event
import math


def rotate_vector(dx: float, dz: float, yaw: float) -> tuple[float, float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    rotated_dx = dx * cos_yaw - dz * sin_yaw
    rotated_dz = dx * sin_yaw + dz * cos_yaw

    return rotated_dx, rotated_dz



@dataclass
class DiscretizedAgentPose:
    idx_x: int
    idx_z: int
    idx_yaw: int
    idx_pitch: int
    yaw_bins: int
    pitch_bins: int

    def __hash__(self) -> int:
        return hash((self.idx_x, self.idx_z, self.idx_yaw, self.idx_pitch))

    def __repr__(self) -> str:
        return f"DiscAgentPose(x={self.idx_x}, z={self.idx_z}, yaw={self.idx_yaw}, h={self.idx_pitch})"

    def get_neigbhoring_states(
        self,
        grid_reachable_positions: list[tuple[int, int]],
        deltas: dict[str, tuple[int,int,int,int] | tuple[float, float, float, float]]
    ) -> list[tuple[str, "DiscretizedAgentPose"]]:
        """Get neighboring states of the given agent pose. Uses gridSnapping for handling diagonal movements"""
        neighbors = []

        for action_name, (dx, dz, dyaw, dpitch) in deltas.items():
            new_idx_yaw = int(self.idx_yaw + math.copysign(dyaw, 1)) % self.yaw_bins 
            new_idx_pitch = int(self.idx_yaw + math.copysign(dpitch, 1))
            
            rotated_dx, rotated_dz = rotate_vector(dx, dz, 360 * self.idx_yaw/self.yaw_bins)
            
            new_idx_x = int(self.idx_x + math.copysign(rotated_dx, 1))
            new_idx_z = int(self.idx_z + math.copysign(rotated_dz, 1))

            if (new_idx_x, new_idx_z) in grid_reachable_positions and 0 <= new_idx_pitch < self.pitch_bins:
                neigbor = DiscretizedAgentPose(
                    idx_x=new_idx_x,
                    idx_z=new_idx_z,
                    idx_yaw=new_idx_yaw,
                    idx_pitch=new_idx_pitch,
                    yaw_bins=self.yaw_bins,
                    pitch_bins=self.pitch_bins
                )
                neighbors.append((action_name, neigbor))

        return neighbors

    def __lt__(self, other: "DiscretizedAgentPose") -> bool:
        return (self.idx_x, self.idx_z, self.idx_yaw, self.idx_pitch) < (
            other.idx_x,
            other.idx_z,
            other.idx_yaw,
            other.idx_pitch,
        )

    def __eq__(self, other: "DiscretizedAgentPose") -> bool:
        return (self.idx_x, self.idx_z, self.idx_yaw, self.idx_pitch) == (
            other.idx_x,
            other.idx_z,
            other.idx_yaw,
            other.idx_pitch,
        )

def get_scene_bounds(controller) -> tuple[float, float, float, float]:
    scene_objects = controller.last_event.metadata["objects"]
    min_x, max_x = float("inf"), float("-inf")
    min_z, max_z = float("inf"), float("-inf")

    for obj in scene_objects:
        try:
            o_min_x = min([v[0] for v in obj["axisAlignedBoundingBox"]["cornerPoints"]])
            o_max_x = max([v[0] for v in obj["axisAlignedBoundingBox"]["cornerPoints"]])
            o_min_z = min([v[2] for v in obj["axisAlignedBoundingBox"]["cornerPoints"]])
            o_max_z = max([v[2] for v in obj["axisAlignedBoundingBox"]["cornerPoints"]])
            min_x = min(min_x, o_min_x)
            max_x = max(max_x, o_max_x)
            min_z = min(min_z, o_min_z)
            max_z = max(max_z, o_max_z)
        except:
            pass
    return (min_x, max_x, min_z, max_z)


def get_grid_bounds(controller, grid_size) -> tuple[float, float, float, float]:

    (min_x, max_x, min_z, max_z) = get_scene_bounds(controller)
    all_pos_reachable2d = list(
        itertools.product(
            np.arange(
                int((grid_size + min_x) / grid_size) * grid_size,
                max_x,
                grid_size,
            ),
            np.arange(
                int((grid_size + min_z) / grid_size) * grid_size,
                max_z,
                grid_size,
            ),
        )
    )

    grid_min_x, grid_max_x = min([c[0] for c in all_pos_reachable2d]), max(
        [c[0] for c in all_pos_reachable2d]
    )
    grid_min_z, grid_max_z = min([c[1] for c in all_pos_reachable2d]), max(
        [c[1] for c in all_pos_reachable2d]
    )
    return (grid_min_x, grid_max_x, grid_min_z, grid_max_z)


def get_dimensions(controller, grid_size) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    int,
    int,
    list[tuple[int, int]],
    list[tuple[int, int]],
]:

    (min_x, max_x, min_z, max_z) = get_scene_bounds(controller)
    (grid_min_x, grid_max_x, grid_min_z, grid_max_z) = get_grid_bounds(
        controller, grid_size
    )

    all_pos_reachable2d = list(
        itertools.product(
            np.arange(
                int((grid_size + min_x) / grid_size) * grid_size,
                max_x,
                grid_size,
            ),
            np.arange(
                int((grid_size + min_z) / grid_size) * grid_size,
                max_z,
                grid_size,
            ),
        )
    )

    grid_cols = int((grid_max_x - grid_min_x) / grid_size) + 1
    grid_rows = int((grid_max_z - grid_min_z) / grid_size) + 1

    out = controller.step(
        action="GetReachablePositions",
        raise_for_failure=True,
    ).metadata["actionReturn"]

    reachable_positions = set(
        (round(p["x"] / grid_size) * grid_size, round(p["z"] / grid_size) * grid_size)
        for p in out
    )

    unreachable_positions = set(all_pos_reachable2d).difference(reachable_positions)

    grid_reachable_positions = [
        (
            round((pos[0] - grid_min_x) / grid_size),
            round((pos[1] - grid_min_z) / grid_size),
        )
        for pos in reachable_positions
    ]
    grid_unreachable_positions = [
        (
            round((pos[0] - grid_min_x) / grid_size),
            round((pos[1] - grid_min_z) / grid_size),
        )
        for pos in unreachable_positions
    ]

    return (
        (min_x, max_x, min_z, max_z),
        (grid_min_x, grid_max_x, grid_min_z, grid_max_z),
        grid_rows,
        grid_cols,
        grid_reachable_positions,
        grid_unreachable_positions,
    )


def teleport_agent_pose(
    controller,
    target_pose: DiscretizedAgentPose,
    grid_size: float,
    grid_bounds: tuple[float, float, float, float],
    min_pitch: float,
    max_pitch: float
) -> None:
    curr_pose = get_discrete_pose(
        controller, 
        grid_size, 
        grid_bounds, 
        yaw_bins=target_pose.yaw_bins, 
        pitch_bins=target_pose.pitch_bins, 
        min_pitch=min_pitch, 
        max_pitch=max_pitch
    )
    y = controller.last_event.metadata["agent"]["position"]["y"]
    (x,z,yaw,pitch) = from_discrete_pose(
        target_pose,
        grid_size,
        grid_bounds,
        min_pitch=min_pitch,
        max_pitch=max_pitch        
    ) 

    if curr_pose != target_pose:
        try:
            controller.step(
                action="TeleportFull",
                position=dict(x=x, y=y, z=z),
                rotation=dict(x=0, y= yaw,z=0),
                standing=True,
                horizon=pitch,
                raise_for_failure=True,
            )
            controller.step(
                action="Pass",
                raise_for_failure=True,
            )
        except Exception as e:
            print(
                f"Teleportation failed at x:{x}, y:{y}, z:{z} with error {str(e)[0:70]}, retrying with forceAction=True"
            )
            controller.step(
                action="TeleportFull",
                position=dict(x=x, y=y, z=z),
                rotation=dict(x=0, y=yaw, z=0),
                standing=True,
                horizon=pitch,
                raise_for_failure=True,
                forceAction=True,
            )
            controller.step(
                action="Pass",
                raise_for_failure=True,
            )

def from_discrete_pose(
    pose: DiscretizedAgentPose,
    grid_size: float, 
    grid_bounds: tuple[float, float, float, float],
    min_pitch: float,
    max_pitch: float
) -> tuple[float, float, float, float]:
    grid_min_x, grid_max_x, grid_min_z, grid_max_z = grid_bounds
    
    x = grid_min_x + pose.idx_x * grid_size
    z = grid_min_z + pose.idx_z * grid_size
    pitch = (pose.idx_pitch/pose.pitch_bins * (max_pitch - min_pitch) + min_pitch)
    yaw  = 360 * pose.idx_yaw / pose.yaw_bins

    return x,z, yaw, pitch


def get_discrete_pose(
    controller, 
    grid_size: float, 
    grid_bounds: tuple[float, float, float, float],
    yaw_bins: int,
    pitch_bins: int,
    min_pitch: float,
    max_pitch: float
) -> DiscretizedAgentPose:
    grid_min_x, grid_max_x, grid_min_z, grid_max_z = grid_bounds
    
    md = controller.last_event.metadata
    agent_x, _, agent_z = tuple(md["agent"]["position"].values())
    _, yaw, _ = tuple(md["agent"]["rotation"].values())
    pitch = md["agent"]["cameraHorizon"]

    return DiscretizedAgentPose(
        idx_x=round((agent_x - grid_min_x) / grid_size),
        idx_z=round((agent_z - grid_min_z) / grid_size),
        idx_yaw=round(yaw_bins * (yaw % 360) / 360),
        idx_pitch=round(pitch_bins * (pitch + min_pitch) / (max_pitch - min_pitch)),
        yaw_bins=yaw_bins,
        pitch_bins=pitch_bins
    )


def get_ground_truth_bbx(
    event: Event, 
    class_names: list[str],
    min_area: float,

) -> list:
    
    def item_is_a_child(obj_id: str) -> bool:
        return "___" in obj_id

    seen_obj_ids = set()
    bounding_boxes = []

    class_names_inv = {name: idx for idx, name in enumerate(class_names)}
    
    visible_objects = {
        obj["objectId"]: obj["visible"] or obj["parentReceptacles"] is None
        for obj in event.metadata["objects"]
    }

    assert event.instance_detections2D is not None
    for objid in event.instance_detections2D:
        if objid in seen_obj_ids:
            continue

        if not visible_objects.get(objid, False):
            continue

        if item_is_a_child(objid):
            continue

        object_class: str = objid.split("|")[0]

        if object_class in class_names_inv:
            class_id = class_names_inv[object_class]

            seen_obj_ids.add(objid)
            xmin, ymin, xmax, ymax = event.instance_detections2D[objid]

            if (xmax - xmin) * (ymax - ymin) < min_area:
                continue

            bounding_boxes.append(
                dict(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    class_id=class_id,
                    class_name=class_names[class_id],
                    bbx_features=np.array([], dtype=float),
                )
            )
    return bounding_boxes



