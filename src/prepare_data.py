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

SIDES = {"left": 1, "right": 2}
REGIONS = { 
    "femur": [13],
    "patella": [14],
    "lowerleg": [15, 16],  # tibia, fibula
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



def filter_poi(poi_object: POI, subject_id: str, side: int, exclude_dict: dict) -> POI:
    """Filter POIs by removing excluded ones for the given subject.
    
    Args:
        poi_object: POI object to filter
        subject_id: Current subject ID
        side: Side of the body (1 for LEFT, 2 for RIGHT)
        exclude_dict: Dictionary of {subject_id: {side: [pois_to_exclude]}}
        
    Returns:
        Filtered POI object
    """

    if subject_id in exclusion_dict and side in exclusion_dict[subject_id]:
        to_remove = exclusion_dict[subject_id][side]
        print(f"→ Entferne {len(to_remove)} POIs für {subject_id}, Seite {side}")
        poi_object = poi_object.remove(*to_remove)
    return poi_object

def rename_poi(poi_object: POI, subject_id: str, side: int) -> POI:
    """Rename POIs based on subject ID and side.

    Args:
        poi_object: POI object to rename
        subject_id: Current subject ID
        side: Side of the body (1 for LEFT, 2 for RIGHT)"""


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
    print(f"Loading POI from: {left_poi_candidate}")

    try:
        poi = POI_Global.load(left_poi_candidate.file["json"])
        # TODO: turn glocal POI into local POI
        return poi
    except Exception as e:
        print(f"Error loading POI: {str(e)}")
        return None

def get_ct(container) -> NII:
    ct_query = container.new_query(flatten=True)
    ct_query.filter_format("ct")
    ct_query.filter_filetype("nii.gz")  # only nifti files
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


def get_bounding_box(mask, leg, margin=1):
    """Get the bounding box of a given leg in a mask.

    Args:
        mask (numpy.ndarray): The mask to search for the vertex.
        vert (int): The vertebra to search for in the mask.
        margin (int, optional): The margin to add to the bounding box. Defaults to 2.

    Returns:
        tuple: A tuple containing the minimum and maximum values for the x, y, and z axes of the
        bounding box.
    """
    indices = np.where(mask == leg)

    #debug
    if len(indices[0]) == 0:
        raise ValueError(f"Vertebra {leg} not found in the mask.")
    
    x_min = np.min(indices[0]) - margin
    x_max = np.max(indices[0]) + margin
    y_min = np.min(indices[1]) - margin
    y_max = np.max(indices[1]) + margin
    z_min = np.min(indices[2]) - margin
    z_max = np.max(indices[2]) + margin

    # Make sure the bounding box is within the mask
    x_min = max(0, x_min)
    x_max = min(mask.shape[0], x_max)
    y_min = max(0, y_min)
    y_max = min(mask.shape[1], y_max)
    z_min = max(0, z_min)
    z_max = min(mask.shape[2], z_max)

    #debug
    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
        raise ValueError(
            f"Invalid bounding box for vertebra {leg}: "
            f"x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}, "
            f"z_min={z_min}, z_max={z_max}"
        )

    return x_min, x_max, y_min, y_max, z_min, z_max


def get_bounding_box(split_mask, subreg_mask, leg_side, region_ids, margin=2):
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

    Returns:
        Tuple[slice, slice, slice]: z, y, x slices to crop the image/volume
    """

    assert split_mask.shape == subreg_mask.shape, "Masks must have the same shape."

    # Create combined mask
    side_mask = split_mask == leg_side
    region_mask = np.isin(subreg_mask, region_ids)
    combined_mask = np.logical_and(side_mask, region_mask)

    # No voxels? Return None or raise
    if not np.any(combined_mask):
        raise ValueError("No voxels found for the given leg side and region.")

    # Get coordinates of the non-zero voxels
    #coords = np.argwhere(combined_mask)
    #zmin, ymin, xmin = coords.min(axis=0) - margin
    #zmax, ymax, xmax = coords.max(axis=0) + margin  # +1 because slicing is exclusive
    indices = np.where(combined_mask)

    x_min = np.min(indices[0]) - margin
    x_max = np.max(indices[0]) + margin
    y_min = np.min(indices[1]) - margin
    y_max = np.max(indices[1]) + margin
    z_min = np.min(indices[2]) - margin
    z_max = np.max(indices[2]) + margin

    # Make sure the bounding box is within the mask
    x_min = max(0, x_min)
    x_max = min(subreg_mask.shape[0], x_max)
    y_min = max(0, y_min)
    y_max = min(subreg_mask.shape[1], y_max)
    z_min = max(0, z_min)
    z_max = min(subreg_mask.shape[2], z_max)
    
    return  x_min, x_max, y_min, y_max, z_min, z_max
    #return (
    #    slice(zmin, zmax),
    #    slice(ymin, ymax),
    #    slice(xmin, xmax)
    #)

def process_container(
    subject,
    container,
    save_path: PathLike,
    rescale_zoom: tuple | None,
    get_files_fn: Callable[[Subject_Container], tuple[POI, POI, NII, NII, NII]],
    exclude: bool = False, #Alissa
):
    print("Subject:", subject)
    right_poi, left_poi, ct, splitseg, subreg = get_files_fn(container)

    
    splitseg.reorient_(("L", "A", "S"))
    subreg.reorient_(("L", "A", "S"))
    right_poi = right_poi.resample_from_to(subreg)
    left_poi = left_poi.resample_from_to(subreg)


    # cut segmentations to left and right
    split_arr = splitseg.get_array()
    subreg_arr = subreg.get_array()

    summary = []

    for leg_side, side_value in SIDES.items():
        if exclude and leg_side == "left":
            left_poi = filter_poi(left_poi, subject, side_value, exclusion_dict)
        elif exclude and leg_side == "right":
            right_poi = filter_poi(right_poi, subject, side_value, exclusion_dict)
        
        for region_name, region_ids in REGIONS.items():

            # get bounding box for the current leg and region
            x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(
                        split_arr, subreg_arr, side_value, region_ids
                    )

            split_path =os.path.join(save_path, region_name, subject, side_value, "split.nii.gz")
            subreg_path = os.path.join(save_path, region_name, subject, side_value, "subreg.nii.gz")
            poi_path = os.path.join(save_path, region_name, subject, side_value, "poi.json")
            poi_path_global = os.path.join(save_path, region_name, subject, side_value, "poi_global.json")

            #create directories if they do not exist
            if not os.path.exists(os.path.join(save_path, subject, str(leg))):
                os.makedirs(os.path.join(save_path, subject, str(leg)))

            try:   
                split_cropped = splitseg.apply_crop(
                    ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
                )
                subreg_cropped = subreg.apply_crop(
                    ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
                )
                if leg_side == "left":
                    left_poi_cropped = left_poi.apply_crop(
                        ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
                    )
                elif leg_side == "right":
                    right_poi_cropped = right_poi.apply_crop(
                        ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
                    )

            except Exception as e:
                print(f"Error processing {subject}: {str(e)}")
                print(f"Crop dimensions: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}, z_min={z_min}, z_max={z_max}")
                print(f"ex_slice: {(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))}")
                #print(f"ct shape: {ct.shape},\n subreg shape: {subreg.shape},\n vertseg shape: {vertseg.shape}, poi shape: {poi.shape}")
                raise
            
            if rescale_zoom:

                #ct_cropped.rescale_(rescale_zoom)
                split_cropped.rescale_(rescale_zoom)
                subreg_cropped.rescale_(rescale_zoom)
                left_poi_cropped.rescale_(rescale_zoom)
                left_poi_cropped.rescale_(rescale_zoom)
    

            #ct_cropped.save(ct_path, verbose=False)
            split_cropped.save(split_path, verbose=False)
            subreg_cropped.save(subreg_path, verbose=False)
            if leg_side == "left":
                left_poi.save(poi_path, verbose=False)
                left_poi.to_global().save_mrk(poi_path_global)
            elif leg_side == "right":
                right_poi.save(poi_path, verbose=False)
                right_poi.to_global().save_mrk(poi_path_global)


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
                    save_path, region_name, subject, side_value, "cutout_slice_indices.json"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(slice_indices, f)

            summary.append(
                {
                    "region": region_name,
                    "subject": subject,
                    "leg": side_value,
                    "file_dir": os.path.join(save_path, region_name, subject, side_value),
                }
            )


    #################################################################################
    """
    for leg in [1, 2]:
        if exclude and leg == 1:
            left_poi = filter_poi(left_poi, subject, leg, exclusion_dict)
        elif exclude and leg == 2:
            right_poi = filter_poi(right_poi, subject, leg, exclusion_dict)

        x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(
                        split_arr, leg
                    )            
        
        split_path = os.path.join(save_path, subject, str(leg), "split.nii.gz")
        subreg_path = os.path.join(save_path, subject, str(leg), "subreg.nii.gz")
        poi_path = os.path.join(save_path, subject, str(leg), "poi.json")
        poi_path_global = os.path.join(save_path, subject, str(leg), "poi_global.json")
        #right_poi_path = os.path.join(save_path, subject, str(leg), "right_poi.json")
        #left_poi_path = os.path.join(save_path, subject, str(leg), "left_poi.json")
        #right_poi_global_path = os.path.join(save_path, subject, str(leg), "right_poi_global.json")
        #left_poi_global_path = os.path.join(save_path, subject, str(leg), "left_poi_global.json")


        #create directories if they do not exist
        if not os.path.exists(os.path.join(save_path, subject, str(leg))):
            os.makedirs(os.path.join(save_path, subject, str(leg)))

        try:            
            #ct_cropped = ct.apply_crop(
            #    ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
            #)
            split_cropped = splitseg.apply_crop(
                ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
            )
            subreg_cropped = subreg.apply_crop(
                ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
            )

        except Exception as e:
            print(f"Error processing {subject}: {str(e)}")
            print(f"Crop dimensions: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}, z_min={z_min}, z_max={z_max}")
            print(f"ex_slice: {(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))}")
            #print(f"ct shape: {ct.shape},\n subreg shape: {subreg.shape},\n vertseg shape: {vertseg.shape}, poi shape: {poi.shape}")
            raise
        
        if rescale_zoom:

            #ct_cropped.rescale_(rescale_zoom)
            split_cropped.rescale_(rescale_zoom)
            subreg_cropped.rescale_(rescale_zoom)
            right_poi.rescale_(rescale_zoom)
            left_poi.rescale_(rescale_zoom)
  

        #ct_cropped.save(ct_path, verbose=False)
        split_cropped.save(split_path, verbose=False)
        subreg_cropped.save(subreg_path, verbose=False)
        if leg == 1:
            left_poi.save(poi_path, verbose=False)
            left_poi.to_global().save_mrk(poi_path_global)
        else:
            right_poi.save(poi_path, verbose=False)
            right_poi.to_global().save_mrk(poi_path_global)


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
                save_path, subject, str(leg), "cutout_slice_indices.json"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(slice_indices, f)

        summary.append(
            {
                "subject": subject,
                "leg": leg,
                "file_dir": os.path.join(save_path, subject, str(leg)),
            }
        )
    """
    
    return summary


def prepare_data(
    bids_surgery_info: BIDS_Global_info,
    save_path: str,
    get_files_fn: callable,
    rescale_zoom: tuple | None = None,
    n_workers: int = 8,
    exclude: bool = False, # Alissa
):
    #master = []
    # This will collect results per region
    region_results = defaultdict(list)

    partial_process_container = partial(
        process_container,
        save_path=save_path,
        rescale_zoom=rescale_zoom,
        get_files_fn=get_files_fn,
        exclude=exclude,  # Pass None if not provided
    )

    """
    master = pqdm(
        bids_surgery_info.enumerate_subjects(),
        partial_process_container,
        n_jobs=n_workers,
        argument_type="args",
        exception_behaviour="immediate",
    )
    master = [item for sublist in master for item in sublist]
    master_df = pd.DataFrame(master)
    master_df.to_csv(os.path.join(save_path, "master_df.csv"), index=False)
    """
    for subject, container in bids_surgery_info.enumerate_subjects():
        result = partial_process_container(subject, container)  # Process sequentially
        #master.extend(result)  # Flatten results (same as your original list comprehension)
        for entry in result:
            region = entry["region"]
            region_results[region].append(entry)

    # Save results for each region
    for region_name, entries in region_results.items():
        df = pd.DataFrame(entries)
        region_folder = os.path.join(save_path, region_name)
        os.makedirs(region_folder, exist_ok=True)
        df.to_csv(os.path.join(region_folder, "master_df.csv"), index=False)
    # Convert to DataFrame and save
    #master_df = pd.DataFrame(master)
    #master_df.to_csv(os.path.join(save_path, "master_df.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()
    print(args.derivatives_name)

    bids_gloabl_info = BIDS_Global_info(
        datasets=[args.data_path], parents=["rawdata", args.derivatives_name]
    )

    get_data_files = partial(
        get_files,
        get_right_poi_fn=get_right_poi,
        get_left_poi_fn=get_left_poi,
        get_ct_fn=get_ct,
        get_splitseg_fn=get_splitseg,
        get_subreg_fn=get_subreg,
    )
    
    prepare_data(
        bids_surgery_info=bids_gloabl_info,
        save_path=args.save_path,
        get_files_fn=get_data_files,
        rescale_zoom=None if args.no_rescale else (0.8, 0.8, 0.8),
        n_workers=args.n_workers,
        exclude=args.exclude, 
    )