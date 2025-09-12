"""Run this before training a model to prepare the data."""

import argparse
import json
import os
from functools import partial
from os import PathLike
from typing import Callable

import numpy as np
import pandas as pd
from collections import defaultdict


from TPTBox import NII, BIDS_Global_info
from TPTBox.core.poi import POI
from TPTBox.core.poi_fun.poi_global import POI_Global
from TPTBox import Subject_Container
from pqdm.processes import pqdm


SIDES = {"left": 1, "right": 2}
REGIONS = { 
    "femur": [13],
    "patella": [14],
    "lowerleg": [15, 16],  # tibia, fibula
}

FOV_POIS = {
    "femur": [11, 12, 13, 19],
    "patella": [8, 9, 10, 14, 15, 16, 17, 18, 20, 21, 22, #femur
                1, 2, 3, 4, 5, 6, 7, # patella

                114, 119, 121, 23, 24, 25, 26, 27, 28, 29, # tibia
                 98, 99,
                
                ], 
    "lowerleg":[30, 31, 32, 120]
}

REGION_POIS = {
    "femur": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "patella": [1, 2, 3, 4, 5, 6, 7],
    "lowerleg": [114, 119, 120, 121, 122, 23, 24, 25, 26, 27, 28, 29, 30, 31, 98, 99, 32]
}


exclusion_dict = {
    "CTFU04045": {
        2: [  # RIGHT
            (15, 27),  # LATERAL_CONDYLE_ANTERIOR → Tibia
            (15, 29),  # LATERAL_CONDYLE_LATERAL → Tibia
            (13, 21),  # MEDIAL_CONDYLE_POSTERIOR → Femur
            (15, 28),  # MEDIAL_CONDYLE_MEDIAL → Tibia
        ]
    },
    "CTFU04656": {
        1: [  # LEFT
            (15, 26),  # MEDIAL_CONDYLE_ANTERIOR → Tibia
        ]
    },
    "MM00024": {
        2: [  # RIGHT
            (15, 28),  # MEDIAL_CONDYLE_MEDIAL → Tibia
            (13, 21),  # MEDIAL_CONDYLE_POSTERIOR → Femur
        ]
    },
    "MM00027": {
        2: [  # RIGHT
            (15, 25),  # LATERAL_INTERCONDYLAR_TUBERCLE → Tibia
            (15, 26),  # MEDIAL_CONDYLE_ANTERIOR → Tibia
            (15, 24),  # MEDIAL_INTERCONDYLAR_TUBERCLE → Tibia
        ]
    },
    "MM00071": {
        1: [  # LEFT
            (13, 11),  # HIP_CENTER → Femur
        ],
        2: [  # RIGHT
            (13, 13),  # TIP_OF_GREATER_TROCHANTER → Femur
        ]
    },
    "MM00214": {
        1: [  # LEFT
            (13, 13),  # TIP_OF_GREATER_TROCHANTER → Femur
        ]
    },
    "MM00233": {
        1: [  # LEFT
            (14, 6),  # PATELLA_RIDGE_DISTAL_POLE → Patella
        ],
        2: [  # RIGHT
            (15, 16),  # LATERAL_CONDYLE_DISTAL → Tibia
            (13, 15),  # LATERAL_CONDYLE_POSTERIOR_CRANIAL → Femur
            (15, 17),  # MEDIAL_CONDYLE_DISTAL → Tibia
            (13, 22),  # MEDIAL_CONDYLE_POSTERIOR_CRANIAL → Femur
        ]
    }
}

exclusion_dict_renamed = {
    "CTFU04045": [(2, 27), (2, 29), (2, 21), (2, 28)],
    "CTFU04656": [(1, 26)],
    "MM00024": [(2, 28), (2, 21)],
    "MM00027": [(2, 25), (2, 26), (2, 24)],
    "MM00071": [(1, 11), (2, 13)],
    "MM00214": [(1, 13)],
    "MM00233": [(1, 6), (2, 16),(2, 15),(2, 17),(2, 22)]
}

# Due to double POI idx -> rename by +100
map_label_double  = {
    (15, 14) : (15, 114),
    (15, 19) : (15, 119),
    (15, 20) : (15, 120),
    (15, 21) : (15, 121)
}

def get_bad_poi_list(exclude: bool, subject_id: str, leg: int,  exclude_dict: dict[str, list[tuple[int, int]]]) ->list[int]:
    """creates the bad poi list for given subject and leg"""

    if exclude:
        bad_pois = exclude_dict.get(subject_id, [])
        filtered_pois = [ poi_id for leg_id, poi_id in bad_pois if leg_id == leg ]
        return filtered_pois
    else:
        return []

def rename_poi(poi_object: POI, leg_id: int) -> POI:
    """renames POI Ids from region id to leg id (1 = left, 2 = right)"""
    
    label_map_full = {key: (leg_id, key[1]) for key in poi_object.keys()}
    poi_object.map_labels(label_map_full=label_map_full, inplace=True)
    return poi_object

def remove_other_region_pois(poi_object: POI, allowed_poi_ids: list[int]) -> POI:
    """removes all POIs that isn't in the given region"""

    pois_to_remove = [
        (leg_id, poi_id)
        for (leg_id, poi_id) in poi_object.centroids.keys()
        if poi_id not in allowed_poi_ids
    ]
    if pois_to_remove:
        poi_object = poi_object.remove(*pois_to_remove)

    return poi_object

def get_right_poi(container) -> POI:
    right_poi_query = container.new_query(flatten=True)
    right_poi_query.filter_format("poi")
    right_poi_query.filter_filetype("json")  # only nifti files
    right_poi_query.filter_self(lambda f: "RIGHT" in str(f.file["json"]).upper())
    
    if not right_poi_query.candidates:
        print("ERROR: No Right POI candidates found!")
        return None
    
    right_poi_candidate = right_poi_query.candidates[0]
    print(f"Loading Right POI from: {right_poi_candidate}")

    try:
        poi = POI_Global.load(right_poi_candidate.file["json"])
        return poi
    except Exception as e:
        print(f"Error loading POI: {str(e)}")
        return None
    
def get_left_poi(container) -> POI:
    left_poi_query = container.new_query(flatten=True)
    left_poi_query.filter_format("poi")
    left_poi_query.filter_filetype("json") 
    left_poi_query.filter_self(lambda f: "LEFT" in str(f.file["json"]).upper())

    if not left_poi_query.candidates:
        print("ERROR: No Left POI candidates found!")
        return None
    
    left_poi_candidate = left_poi_query.candidates[0]
    print(f"Loading Left POI from: {left_poi_candidate}")

    try:
        poi = POI_Global.load(left_poi_candidate.file["json"])
        return poi
    except Exception as e:
        print(f"Error loading POI: {str(e)}")
        return None

def get_ct(container) -> NII:
    ct_query = container.new_query(flatten=True)
    ct_query.filter_format("ct")
    ct_query.filter_filetype("nii.gz")  # only nifti files

    if not ct_query.candidates:
        print("ERROR: no CT candidates found!")
        return None
        
    ct_candidate = ct_query.candidates[0]

    try:
        ct = ct_candidate.open_nii()
        return ct
    except Exception as e:
        print(f"Error opening CT: {str(e)}")
        return None  

def get_splitseg(container) -> NII:
    splitseg_query = container.new_query(flatten=True)
    splitseg_query.filter_format("split")
    splitseg_query.filter_filetype("nii.gz")  # only nifti files
    #splitseg_query.filter("seg", "subreg")
    if not splitseg_query.candidates:
        print("ERROR: No splitseg candidates found!")
        return None
    splitseg_candidate = splitseg_query.candidates[0]

    try:
        splitseg = splitseg_candidate.open_nii()
        return splitseg
    except Exception as e:
        print(f"Error opening splitseg: {str(e)}")
        return None

def get_subreg(container) -> NII:
    subreg_query = container.new_query(flatten=True)
    subreg_query.filter_format("msk")
    subreg_query.filter_filetype("nii.gz")  # only nifti files
    if not subreg_query.candidates:
        print("ERROR: No subreg candidates found!")
        return None

    subreg_candidate = subreg_query.candidates[0]

    try:
        subreg = subreg_candidate.open_nii()
        return subreg
    except Exception as e:
        print(f"Error opening subreg: {str(e)}")
        return None

def get_files(
    container,
    get_right_poi_fn: Callable,
    get_left_poi_fn: Callable,
    get_ct_fn: Callable,
    get_splitseg_fn: Callable,
    get_subreg_fn: Callable,
) -> tuple[POI, POI, NII, NII, NII]:
    return (
        get_right_poi_fn(container),
        get_left_poi_fn(container),
        get_ct_fn(container),
        get_splitseg_fn(container),
        get_subreg_fn(container),
    )

def get_bounding_box_region(split_mask, subreg_mask, leg_side, region_ids, straight_cut, margin=2):
    """
    Computes a bounding box that covers the region defined by:
    - a specific leg side (from split_mask)
    - and one or more anatomical regions (from subregion_mask)

    Args:
        split_mask (np.ndarray): Mask that contains side information (e.g., 1=left, 2=right)
        subreg_mask (np.ndarray): Mask that contains anatomical region info
        leg_side (int): The side of the leg to extract (1=left, 2=right)
        region_ids (List[int]): List of subregion IDs to include (e.g., [13] for femur)
        margin (int): Margin to add around the bounding box
        straight_cut (bool): If set, horizontally cut across the entire leg for contextual input, 
        otherwise cut into regions

    Returns:
        Tuple[int, int, int, int, int, int]: x_min, x_max, y_min, y_max, z_min, z_max
    """

    assert split_mask.shape == subreg_mask.shape, "Masks must have the same shape."

    # Create combined mask
    side_mask = split_mask == leg_side
    region_mask = np.isin(subreg_mask, region_ids)
    combined_mask = np.logical_and(side_mask, region_mask)

    if not np.any(combined_mask):
        raise ValueError("No voxels found for the given leg side and region.")

    indices = np.where(combined_mask)

    if straight_cut:
        print("STRAIGHT_CUT (bounding box)")
        full_leg_indices = np.where(side_mask)
        x_min = np.min(full_leg_indices[0]) - margin
        x_max = np.max(full_leg_indices[0]) + margin
        y_min = np.min(full_leg_indices[1]) - margin
        y_max = np.max(full_leg_indices[1]) + margin

        z_min = np.min(indices[2]) - margin
        z_max = np.max(indices[2]) + margin
    else:
        x_min = np.min(indices[0]) - margin
        x_max = np.max(indices[0]) + margin
        y_min = np.min(indices[1]) - margin
        y_max = np.max(indices[1]) + margin
        z_min = np.min(indices[2]) - margin
        z_max = np.max(indices[2]) + margin

    x_min = max(0, x_min)
    x_max = min(subreg_mask.shape[0], x_max)
    y_min = max(0, y_min)
    y_max = min(subreg_mask.shape[1], y_max)
    z_min = max(0, z_min)
    z_max = min(subreg_mask.shape[2], z_max)
    
    return  x_min, x_max, y_min, y_max, z_min, z_max


def get_bounding_box_fov(split_mask, subreg_mask, leg_side, fov_pois, poi, margin=5):
    """
    Computes a bounding box for the 3 FOVs (Upper Femur, Lower Femur + Patella + Upper Lower Leg, Lower Lower Leg)

    Args:
        split_mask (np.ndarray): Mask that contains side information (e.g., 1=left, 2=right)
        subreg_mask (np.ndarray): Mask that contains anatomical region info
        leg_side (int): The side of the leg to extract (1=left, 2=right)
        fov_pois (list): List containing POI ids that should be included in the FOV (bounding box) (eg. [1, 2, 3, 4, 5] )
        poi (POI): POI object
        margin (int): Margin to add around the bounding box

    Returns:
        Tuple[int, int, int, int, int, int]: x_min, x_max, y_min, y_max, z_min, z_max
    """


    assert split_mask.shape == subreg_mask.shape, "Masks must have the same shape."

    # Create combined mask
    side_mask = split_mask == leg_side

    # get all coordinates of the POIs in the FOV
    fov_coords = np.array([poi.centroids[(leg_side, poi_id)] for poi_id in fov_pois])

    # retrieve the min and max z-coordinates of the FOV POIs

    z_min = int(np.floor(np.min(fov_coords[:, 2])))
    z_min -= margin
    z_max = int(np.ceil(np.max(fov_coords[:, 2]))) 
    z_max += margin
    
    full_leg_indices = np.where(side_mask)
    x_min = np.min(full_leg_indices[0]) - margin
    x_max = np.max(full_leg_indices[0]) + margin
    y_min = np.min(full_leg_indices[1]) - margin
    y_max = np.max(full_leg_indices[1]) + margin

    x_min = max(0, x_min)
    x_max = min(subreg_mask.shape[0], x_max)
    y_min = max(0, y_min)
    y_max = min(subreg_mask.shape[1], y_max)
    z_min = max(0, z_min)
    z_max = min(subreg_mask.shape[2], z_max)
    
    return  x_min, x_max, y_min, y_max, z_min, z_max

def get_bounding_box_side(split_mask, leg_side, margin=2):
    """
    Computes a bounding box for the entire leg side (left or right) based on the split_mask.

    Args:
        split_mask (np.ndarray): Mask that contains side information (e.g., 1=left, 2=right)
        subreg_mask (np.ndarray): Mask that contains anatomical region info
        leg_side (int): The side of the leg to extract (1=left, 2=right)

    Returns:
        Tuple[int, int, int, int, int, int]: x_min, x_max, y_min, y_max, z_min, z_max
    """

    # Create combined mask
    side_mask = split_mask == leg_side

    # Get coordinates of the non-zero voxels
    indices = np.where(side_mask)
    x_min = np.min(indices[0]) - margin
    x_max = np.max(indices[0]) + margin
    y_min = np.min(indices[1]) - margin
    y_max = np.max(indices[1]) + margin
    z_min = np.min(indices[2]) - margin
    z_max = np.max(indices[2]) + margin

    x_min = max(0, x_min)
    x_max = min(split_mask.shape[0], x_max)
    y_min = max(0, y_min)
    y_max = min(split_mask.shape[1], y_max)
    z_min = max(0, z_min)
    z_max = min(split_mask.shape[2], z_max)
    
    return  x_min, x_max, y_min, y_max, z_min, z_max

def cut_and_save(save_cut_path, splitseg, subreg, ct, poi, x_min, x_max, y_min, y_max, z_min, z_max, rescale_zoom=None, region_pois=None):
    """Crops and saves the data files according to the bounding box."""

    split_path = os.path.join(save_cut_path, "split.nii.gz")
    subreg_path = os.path.join(save_cut_path, "subreg.nii.gz")
    ct_path = os.path.join(save_cut_path, "ct.nii.gz")
    poi_path = os.path.join(save_cut_path, "poi.json")
    poi_path_global = os.path.join(save_cut_path, "poi_global.json")

    #create directories if they do not exist
    if not os.path.exists(os.path.join(save_cut_path)):
        os.makedirs(os.path.join(save_cut_path))
    try:   
        split_cropped = splitseg.apply_crop(
            ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
        )
        subreg_cropped = subreg.apply_crop(
            ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
        )
        ct_cropped = ct.apply_crop(
                    ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
                )
        
        poi_cropped = poi.apply_crop(
                o_shift=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
            )
        poi_cropped.filter_points_inside_shape(inplace=True)

        if region_pois:
            poi_cropped = remove_other_region_pois(poi_cropped, region_pois)

    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Crop dimensions: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}, z_min={z_min}, z_max={z_max}")
        print(f"ex_slice: {(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))}")
        raise
    
    if rescale_zoom:
        split_cropped.rescale_(rescale_zoom)
        subreg_cropped.rescale_(rescale_zoom)
        ct_cropped.rescale(rescale_zoom)
        poi_cropped.rescale_(rescale_zoom)

    split_cropped.save(split_path, verbose=False)
    subreg_cropped.save(subreg_path, verbose=False)
    ct_cropped.save(ct_path, verbose=False)
    poi_cropped.save(poi_path, verbose=False)
    poi_cropped.to_global().save_mrk(poi_path_global)

    # Save the slice indices as json to reconstruct the original POI file (there probably is a more BIDS-like approach to this)
    slice_indices = {
        "x_min": int(x_min),
        "x_max": int(x_max),
        "y_min": int(y_min),
        "y_max": int(y_max),
        "z_min": int(z_min),
        "z_max": int(z_max),
    }
    with open(
        os.path.join(
            save_cut_path, "cutout_slice_indices.json"
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(slice_indices, f)


def process_container(
    subject,
    container,
    save_path: PathLike,
    rescale_zoom: tuple | None,
    get_files_fn: Callable[[Subject_Container], tuple[POI, POI, NII, NII, NII]],
    exclude: bool = False, 
    straight_cut: bool = False,
    fov_cut: bool = False,
    side_cut: bool = False
):
    right_poi, left_poi, ct, splitseg, subreg = get_files_fn(container)

    if any(x is None for x in (right_poi, left_poi, splitseg, subreg)):
        print("SKIP!")
        return []

    splitseg.reorient_(("L", "A", "S"))
    subreg.reorient_(("L", "A", "S"))
    ct.reorient_(("L", "A", "S"))
    right_poi = right_poi.resample_from_to(subreg)
    left_poi = left_poi.resample_from_to(subreg)

    # rename double POI ids
    right_poi = right_poi.map_labels(label_map_full=map_label_double, inplace=True)
    left_poi = left_poi.map_labels(label_map_full=map_label_double, inplace=True)
    # rename POI ids to match the leg id -> needed for training model
    right_poi = rename_poi(right_poi, 2)
    left_poi = rename_poi(left_poi, 1)

    # cut segmentations to left and right
    split_arr = splitseg.get_array()
    subreg_arr = subreg.get_array()

    summary = []

    for leg_side, leg_id in SIDES.items():      

        if side_cut:
            # get bounding box for the entire leg side
            print("side_cut before getbb:", side_cut)
            x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box_side(
                split_arr, leg_id
            )  
            save_cut_path = os.path.join(save_path, subject, leg_side)
            cut_and_save(save_cut_path, splitseg, subreg, ct, right_poi if leg_side == "right" else left_poi, x_min, x_max, y_min, y_max, z_min, z_max, rescale_zoom)
            summary.append(
                {
                    "subject": subject,
                    "leg": leg_id,
                    "file_dir": save_cut_path,
                    "bad_poi_list": get_bad_poi_list(exclude, subject, leg_id, exclusion_dict_renamed)
                })

        else:
            for region_name, region_ids in REGIONS.items():
                save_cut_path = os.path.join(save_path, region_name, subject, leg_side)

                if fov_cut:
                    # get bounding box for the FOV POIs
                    print("fov_cut before getbb:", fov_cut)
                    x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box_fov(
                                split_arr, subreg_arr, leg_id, FOV_POIS[region_name], right_poi if leg_side == "right" else left_poi
                            )
                    cut_and_save(save_cut_path, splitseg, subreg, ct, right_poi if leg_side == "right" else left_poi, x_min, x_max, y_min, y_max, z_min, z_max, rescale_zoom)

                else:
                    # get bounding box for the current leg and region
                    print("straight_cut before getbb:", straight_cut)
                    x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box_region(
                                split_arr, subreg_arr, leg_id, region_ids, straight_cut
                            )
                    cut_and_save(save_cut_path, splitseg, subreg, ct, right_poi if leg_side == "right" else left_poi, x_min, x_max, y_min, y_max, z_min, z_max, rescale_zoom, REGION_POIS[region_name])

                summary.append(
                    {
                        "region": region_name,
                        "subject": subject,
                        "leg": leg_id,
                        "file_dir": save_cut_path,
                        "bad_poi_list": get_bad_poi_list(exclude, subject, leg_id, exclusion_dict_renamed)
                    }
                )
    
    return summary


def prepare_data(
    bids_surgery_info: BIDS_Global_info,
    save_path: str,
    get_files_fn: callable,
    rescale_zoom: tuple | None = None,
    n_workers: int = 8,
    exclude: bool = False,
    straight_cut: bool = False,
    fov_cut: bool = False,
    side_cut: bool = False
):
    region_results = defaultdict(list)

    partial_process_container = partial(
        process_container,
        save_path=save_path,
        rescale_zoom=rescale_zoom,
        get_files_fn=get_files_fn,
        exclude=exclude,  
        straight_cut=straight_cut,
        fov_cut=fov_cut,
        side_cut=side_cut
    )

    
    master = pqdm(
        bids_surgery_info.enumerate_subjects(),
        partial_process_container,
        n_jobs=n_workers,
        argument_type="args",
        exception_behaviour="immediate",
    )
    master = [item for sublist in master for item in sublist]
    
    if not side_cut:
        # group by region
        region_results = defaultdict(list)
        for entry in master:
            region = entry["region"]
            region_results[region].append(entry)

        # save each region
        for region_name, entries in region_results.items():
            df = pd.DataFrame(entries)
            region_folder = os.path.join(save_path, region_name)
            os.makedirs(region_folder, exist_ok=True)
            df.to_csv(os.path.join(region_folder, "master_df.csv"), index=False)

    else:
        # if side_cut=True -> save into one file
        master_df = pd.DataFrame(master)
        os.makedirs(save_path, exist_ok=True)
        master_df.to_csv(os.path.join(save_path, "master_df.csv"), index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cut_group = parser.add_mutually_exclusive_group(required=False)

    parser.add_argument(
        "--data_path", type=str, help="The path to the BIDS dataset", required=True
    )
    parser.add_argument(
        "--derivatives_name",
        type=str,
        help="The name of the derivatives folder",
        required=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="The path to save the prepared data",
        required=True,
    )
    parser.add_argument(
        "--no_rescale",
        action="store_true",
        help="Whether to skip rescaling the data to isotropic voxels",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        help="The number of workers to use for parallel processing",
        default=8,
    )

    parser.add_argument(
        '--exclude',
        action="store_true",
        help='Whether to exclude certain POIs based on a predefined dictionary',
    )

    cut_group.add_argument(
        '--straight_cut',
        action="store_true",
        help='Whether cut the data straight across the leg for contextual input, otherwise cut into regions',
    )

    cut_group.add_argument(
        '--fov_cut',
        action="store_true",
        help='Whether to cut the data based on the FOV POIs (upper femur, lower femur + patella + upper lower leg, lower lower leg)',
    )

    cut_group.add_argument(
        '--side_cut',
        action="store_true",
        help='Whether to cut the data based on the side of the leg (left or right)',
    )

    args = parser.parse_args()
    print(args.derivatives_name)
    print(args.data_path)
    print(args.save_path)

    bids_global_info = BIDS_Global_info(
        datasets=[args.data_path], parents=["rawdata", args.derivatives_name]
    )

    print("BIDS Global Info:", bids_global_info)

    subjects = list(bids_global_info.enumerate_subjects())
    print("Subjects found:", subjects)
    get_data_files = partial(
        get_files,
        get_right_poi_fn=get_right_poi,
        get_left_poi_fn=get_left_poi,
        get_ct_fn=get_ct,
        get_splitseg_fn=get_splitseg,
        get_subreg_fn=get_subreg,
    )
    
    prepare_data(
        bids_surgery_info=bids_global_info,
        save_path=args.save_path,
        get_files_fn=get_data_files,
        rescale_zoom=None if args.no_rescale else (1.5, 1.5, 1.5),#(0.8, 0.8, 0.8),
        n_workers=args.n_workers,
        exclude=args.exclude, 
        straight_cut=args.straight_cut,
        fov_cut=args.fov_cut,
        side_cut=args.side_cut
    )