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
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader, MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

from common.hssd_od_open_voc.hssd_open_voc_env import HSSD_OpenVoc_Env
from common.utils.plot_utils import plot_semantic_2d_map, make_mosaic
from common.utils.data_utils import save_img
from common.vision.detic import build_detic_predictor

import habitat_od.od_dataset_registry
from habitat_active_od.agent import ActiveODAgent

def overlap(mask1: np.ndarray,mask2:np.ndarray):
    return np.logical_and(mask1, mask2).any()

if __name__ == "__main__":
    config = habitat.get_config(config_path="config/habitat_active_od_config.yaml")
    print(OmegaConf.to_yaml(config))

    rng_gen = np.random.default_rng(0)

    habitat_env = HSSD_OpenVoc_Env(config=config)
    habitat_env.reset()

    raise ValueError
    # agent = ActiveODAgent(config=config)
    # viewpoint_dataset = PointNavDatasetV1(config.HABITAT_ACTIVE_OD.viewpoint_dataset.dataset)

    # scene_names = habitat_env.get_scenes_names()
    # metadata = MetadataCatalog.get("hssd_od_openvoc_test")

    # # detic_config = OmegaConf.load("config/detic_config.yaml")
    # # detic_predictor = build_detic_predictor(detic_config, habitat_env.get_classes()) # type: ignore

    # class_mapping = habitat_env.get_class_mapping()

    # scene_id = None

    # for episode in tqdm(viewpoint_dataset.episodes):
    #     ep_scene_id = episode.scene_id.split('/')[-1]

    #     if len(episode.info["viewpoints"])<8:
    #         continue
    
    #     if ep_scene_id != scene_id:
    #         scene_id = ep_scene_id
    #         habitat_env.change_scene(scene_id)
    #         object2class = habitat_env.get_scene_annotations()
        
    #     agent.reset()

    #     object_id = int(episode.episode_id.split("_obj_id_")[-1])
    #     class_name = object2class[object_id]
        
    #     agent_state = AgentState(
    #         position=episode.start_position,
    #         rotation=episode.start_rotation
    #     )

    #     obs, labels = habitat_env.get_obs_gt(agent_state)
    #     habitat_env.set_goal_image(obs_rgb=obs.rgb)

    #     agent.act(obs)