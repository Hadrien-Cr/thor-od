import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

import habitat # type: ignore
from habitat_sim.agent.agent import AgentState
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1, NavigationEpisode, NavigationGoal # type: ignore
from detectron2.utils.visualizer import Visualizer as DetVisualizer
from detectron2.data import build_detection_test_loader, MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

from common.hssd_od_open_voc.hssd_open_voc_env import HSSD_OpenVoc_Env
from common.utils.plot_utils import plot_semantic_2d_map, make_mosaic
from common.utils.data_utils import save_img
from common.vision.detic import build_detic_predictor

import habitat_od.od_dataset_registry
from agent import ActiveODAgent, DiscreteNavigationAction

metadata = MetadataCatalog.get("hssd_od_openvoc_test")


if __name__ == "__main__":
    config = habitat.get_config(config_path="config/habitat_active_od_config.yaml")
    print(OmegaConf.to_yaml(config))

    rng_gen = np.random.default_rng(0)

    habitat_env = HSSD_OpenVoc_Env(config=config)
    habitat_env.reset()
    obs, labels = habitat_env.get_obs_gt(habitat_env.get_agent_state(), 0)

    agent = ActiveODAgent(config=config)
    agent.reset()

    metadata = MetadataCatalog.get("hssd_od_openvoc_test")

    ACTIONS = {
        DiscreteNavigationAction.MOVE_FORWARD: "move_forward",
        DiscreteNavigationAction.TURN_LEFT: "turn_left",
        DiscreteNavigationAction.TURN_RIGHT: "turn_right",
        DiscreteNavigationAction.STOP: "stop",
    }

    for t in range(100):
        agent_state = habitat_env.get_agent_state()
        rot_x, rot_y, rot_z, rot_w = agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z, agent_state.rotation.w

        action = agent.act(obs)
        habitat_env.step(ACTIONS[action])
        obs, labels = habitat_env.get_obs_gt(habitat_env.get_agent_state(), t)

    # # detic_config = OmegaConf.load("config/detic_config.yaml")
    # # detic_predictor = build_detic_predictor(detic_config, habitat_env.get_classes()) # type: ignore

    # class_mapping = habitat_env.get_class_mapping()

    # scene_id = None

    # viewpoints = habitat_env._current_episode.info["viewpoints"]

    # habitat_obj_occupancy_grid = habitat_env.get_oracle_object_occupancy_grid(0.25)
    # object_annotations = habitat_env.get_object_annotations()

    # object_id = int(habitat_env._current_episode.episode_id.split("_obj_id_")[-1])
    # class_name = object_annotations[object_id]

    # for i, vp in enumerate(viewpoints):
    #     agent_state = AgentState(
    #         position=vp["position"],
    #         rotation=vp["rotation"]
    #     )

    #     obs, labels = habitat_env.get_obs_gt(agent_state)

    #     target_masks = [inst["mask"] for inst in labels.instances if inst["object_id"] == object_id]
    #     det_visualizer = DetVisualizer(
    #         obs.rgb,
    #         metadata=metadata,
    #         scale=0.5,
    #         instance_mode=ColorMode.SEGMENTATION
    #     )

    #     for target_mask in target_masks:
    #         det_visualizer.draw_binary_mask(target_mask)

    #     o = det_visualizer.get_output().get_image()
    #     Image.fromarray(o).save(f"vp/vp{i}.png")