import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

import habitat # type: ignore
from habitat.config import read_write # type: ignore
from habitat.config.default import get_agent_config # type: ignore

from common.hssd_od_open_voc.hssd_open_voc_env import HSSD_OpenVoc_Env
from common.utils.data_utils import save_img, delete_image, agent_state2fname
from common.utils.dataset_utils import save_dataset
from common.utils.plot_utils import plot_semantic_2d_map, make_mosaic
from common.utils.sampling_utils import area_bin_sampling


if __name__ == "__main__": 
    config = habitat.get_config(config_path="habitat_od/config/data_gen_config.yaml")

    if os.path.exists(Path(config.DATA_GEN.dataset.data_root) / config.DATA_GEN.dataset.dataset_name):
        raise ValueError("Change dataset_name so you dont overwrite previous")
    
    rng_gen = np.random.default_rng(0)

    habitat_env = HSSD_OpenVoc_Env(config=config)
    scene_names = habitat_env.get_scenes_names()
    
    candidates_samples = []
    per_class_object_occurences = defaultdict(int)
    class_mapping = habitat_env.get_class_mapping()

    for i, scene in enumerate(scene_names[0:config.DATA_GEN.num_scenes]):
        habitat_env.change_scene(scene)

        print("-----------------")
        print("Collection in Scene = ", scene, f"({i}/{len(scene_names)})")

        habitat_obj_occupancy_grid = habitat_env.get_oracle_object_occupancy_grid(config.DATA_GEN.meters_per_grid_pixel)
        object2class = habitat_env.get_object_annotations()

        for object_id, class_name in object2class.items():
            if class_name == "unknown":
                continue
        
            per_class_object_occurences[class_name] += 1
            
            candidate_agent_states = habitat_obj_occupancy_grid.get_all_viewpoints(object_id, viewpoint_spacing=config.DATA_GEN.viewpoint_spacing)
            rng_gen.shuffle(candidate_agent_states) # type: ignore
            candidate_agent_states = candidate_agent_states[0:config.DATA_GEN.num_samples // 4]

            if not candidate_agent_states:
                continue
            
            for agent_state in tqdm(candidate_agent_states, desc=class_name):
                obs, labels = habitat_env.get_obs_gt(agent_state)

                if not sum(inst["class_name"] == class_name and inst["mask_area"] >= config.DATA_GEN.min_pixel_area for inst in labels.instances):
                    continue
                
                fname = agent_state2fname(
                    "cls_" + class_name  + 
                    "_habitat_scene_" + scene 
                    + "_objid_" + str(object_id), 
                    agent_state
                )

                save_img(
                    obs.rgb, 
                    Path(config.DATA_GEN.dataset.data_root) / config.DATA_GEN.dataset.dataset_name /"test", 
                    fname=fname
                )
                candidates_samples.append((fname, labels.instances))

    rng_gen.shuffle(candidates_samples)
    splits = {
        "test": []
    }

    for class_name in class_mapping:
        # For each class, performs a downsampling to reach "num_samples"
        per_class_candidate_samples = [
            (fname, instances) for (fname, instances) in candidates_samples if ("cls_" + class_name + "_habitat_scene_") in str(fname)
        ]
        if not per_class_candidate_samples:
            continue

        selected_indices = area_bin_sampling(
            per_class_candidate_samples,
            rng_gen,
            mask_filtering_fn=lambda m: (m["class_name"] == class_name),
            num_samples=config.DATA_GEN.num_samples,
        ) 
        assert len(selected_indices) <= config.DATA_GEN.num_samples

        selected_samples = [per_class_candidate_samples[i] for i in selected_indices]
        rejected_samples = [per_class_candidate_samples[i] for i in range(len(per_class_candidate_samples)) if i not in selected_indices]
        
        for (fname, instances) in rejected_samples:
            delete_image(data_dir=Path(config.DATA_GEN.dataset.data_root) / config.DATA_GEN.dataset.dataset_name /"test", fname = fname)

        if selected_samples:
            splits["test"].extend(selected_samples)

    k = len(per_class_object_occurences)
    occ = sorted(per_class_object_occurences.values())
    one_third_value, two_third_value = occ[k//3], occ[(2*k)//3]
    
    class_frequency_mapping = {
        name: "rare" if f <= one_third_value else ("common" if f <= two_third_value else "frequent")
        for name, f in per_class_object_occurences.items()
    }

    for class_name in class_mapping:
        if class_name not in class_frequency_mapping:
            class_frequency_mapping[class_name] = "rare"

    save_dataset(
        config.DATA_GEN, 
        splits, 
        habitat_env.get_class_mapping(),
        class_frequency_mapping
    )