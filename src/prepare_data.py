"""Run this before training a model to prepare the data."""

import argparse
import json
import os
from functools import partial
from os import PathLike
from typing import Callable

import numpy as np
import pandas as pd

from TPTBox import NII, BIDS_Global_info
from TPTBox.core.poi import POI
from TPTBox.core.poi_fun.poi_global import POI_Global
from TPTBox import Subject_Container



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
        print("ERROR: No POI candidates found!")
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


def get_bounding_box(mask, vert, margin=5):
    """Get the bounding box of a given vertebra in a mask.

    Args:
        mask (numpy.ndarray): The mask to search for the vertex.
        vert (int): The vertebra to search for in the mask.
        margin (int, optional): The margin to add to the bounding box. Defaults to 2.

    Returns:
        tuple: A tuple containing the minimum and maximum values for the x, y, and z axes of the
        bounding box.
    """
    indices = np.where(mask == vert)

    #debug
    if len(indices[0]) == 0:
        raise ValueError(f"Vertebra {vert} not found in the mask.")
    
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
            f"Invalid bounding box for vertebra {vert}: "
            f"x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}, "
            f"z_min={z_min}, z_max={z_max}"
        )

    return x_min, x_max, y_min, y_max, z_min, z_max


def process_container(
    subject,
    container,
    save_path: PathLike,
    rescale_zoom: tuple | None,
    get_files_fn: Callable[[Subject_Container], tuple[POI, POI, NII, NII, NII]],
    exclude: bool = False, #Alissa
    flip_to_right: bool = False,
):
    right_poi, left_poi, ct, splitseg, subreg = get_files_fn(container)

    #if exclusion_dict is not None:
    #    poi = filter_poi(poi, f"sub-{subject}", exclusion_dict)
    
    #reorient data to same orientation
    #ct.reorient_(("L", "A", "S"))
    splitseg.reorient_(("L", "A", "S"))
    subreg.reorient_(("L", "A", "S"))
    right_poi = right_poi.resample_from_to(subreg)
    left_poi = left_poi.resample_from_to(subreg)


    # cut segmentations to left and right
    split_arr = splitseg.get_array()

    summary = []

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
        

        if flip_to_right and leg == 1:
            print("Flipping to right leg orientation")
            print("split_cropped affine (before reorient): \n", split_cropped.affine)
            print("subreg_cropped affine (before reorient): \n", subreg_cropped.affine)
            split_cropped.reorient_(("R", "A", "S"))
            subreg_cropped.reorient_(("R", "A", "S"))
            left_poi.reorient_(("R", "A", "S"))
            print("split_cropped affine (after reorient): \n", split_cropped.affine)
            print("subreg_cropped affine (after reorient): \n", subreg_cropped.affine)



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
    vertebrae = {key[0] for key in poi.keys()} 
    vertseg_arr = vertseg.get_array() 
    summary = []

    print("process container: included neighbouring vertebrae: ", include_neighbouring_vertebrae)

    #for vert in vertebrae: #loops through each vertebra ID (extracted from POI keys)
    vertebrae = sorted(vertebrae)
    for index in range(len(vertebrae)): #loops through each vertebra ID (extracted from POI keys)
        vert = vertebrae[index]  
        if vert in vertseg_arr: #vertebra found in segmentation mask
            
            #TODO: muss ich schauen ob die nachbarn in vertseg_arr sind? wenn nicht was dann?
            if include_neighbouring_vertebrae:
                vert_neighbours = [vert]
                if index > 0:
                    vert_neighbours.insert(0, vertebrae[index - 1])
                if index < len(vertebrae) - 1:
                    vert_neighbours.append(vertebrae[index + 1])
                
                print(f"Vertebra {vert} neighbours: {vert_neighbours}")

                # Initialize bounding box limits
                x_min, x_max = np.inf, -np.inf
                y_min, y_max = np.inf, -np.inf
                z_min, z_max = np.inf, -np.inf    

                for v in vert_neighbours:
                    try:
                        bounds = get_bounding_box(vertseg_arr, v)
                    except ValueError as e:
                        print(f"Error getting bounding box for vertebra {v}: {str(e)}")
                        continue
                    
                    x_min = min(x_min, bounds[0])
                    x_max = max(x_max, bounds[1])
                    y_min = min(y_min, bounds[2])
                    y_max = max(y_max, bounds[3])
                    z_min = min(z_min, bounds[4])
                    z_max = max(z_max, bounds[5])
               

            else:
                try:
                    x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(
                        vertseg_arr, vert
                    )
                except ValueError as e:
                    print(f"Error getting bounding box for vertebra {vert}: {str(e)}")
                    continue

            #defines output paths for cropped files
            #ct_path = os.path.join(save_path, subject, str(vert), "ct.nii.gz")
            subreg_path = os.path.join(save_path, subject, str(vert), "subreg.nii.gz")
            vertseg_path = os.path.join(save_path, subject, str(vert), "vertseg.nii.gz")
            poi_path = os.path.join(save_path, subject, str(vert), "poi.json")

            #create directories if they do not exist
            if not os.path.exists(os.path.join(save_path, subject, str(vert))):
                os.makedirs(os.path.join(save_path, subject, str(vert)))

            try:            
                #ct_cropped = ct.apply_crop(
                #    ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
                #)
                subreg_cropped = subreg.apply_crop(
                    ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
                )
                vertseg_cropped = vertseg.apply_crop(
                    ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
                )
                poi_cropped = poi.apply_crop(
                    o_shift=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
                )

            except Exception as e:
                print(f"Error processing {subject}: {str(e)}")
                print(f"Crop dimensions: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}, z_min={z_min}, z_max={z_max}")
                print(f"ex_slice: {(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))}")
                #print(f"ct shape: {ct.shape},\n subreg shape: {subreg.shape},\n vertseg shape: {vertseg.shape}, poi shape: {poi.shape}")
                raise
            
            if rescale_zoom:

                #ct_cropped.rescale_(rescale_zoom)
                subreg_cropped.rescale_(rescale_zoom)
                vertseg_cropped.rescale_(rescale_zoom)
                poi_cropped.rescale_(rescale_zoom)

            #ALISSA: CHECK
            if vert in [11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 24]:
                if (vert, 101) in poi.centroids:
                    print(f"subject: {subject}, Centroid ({vert}, 101) vorhanden!")
                else:
                    print(f"subject: {subject}, Centroid ({vert}, 101) NICHT vorhanden!")


            #ct_cropped.save(ct_path, verbose=False)
            subreg_cropped.save(subreg_path, verbose=False)
            vertseg_cropped.save(vertseg_path, verbose=False)
            #print(f"poi_cropped: \n{poi_cropped.centroids}")
            poi_cropped.save(poi_path, verbose=False)

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
                    save_path, subject, str(vert), "cutout_slice_indices.json"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(slice_indices, f)

            summary.append(
                {
                    "subject": subject,
                    "vertebra": vert,
                    "file_dir": os.path.join(save_path, subject, str(vert)),
                }
            )

        else:
            print(f"Vertebra {vert} has no segmentation for subject {subject}")
    """
    return summary


def prepare_data(
    bids_surgery_info: BIDS_Global_info,
    save_path: str,
    get_files_fn: callable,
    rescale_zoom: tuple | None = None,
    n_workers: int = 8,
    exclude: bool = False, # Alissa

    flip_to_right: bool = False,  # Alissa
):
    master = []

    partial_process_container = partial(
        process_container,
        save_path=save_path,
        rescale_zoom=rescale_zoom,
        get_files_fn=get_files_fn,
        exclude=exclude,  # Pass None if not provided
        flip_to_right=flip_to_right,  # Alissa
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
        master.extend(result)  # Flatten results (same as your original list comprehension)

    # Convert to DataFrame and save
    master_df = pd.DataFrame(master)
    master_df.to_csv(os.path.join(save_path, "master_df.csv"), index=False)


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
    
    """parser.add_argument(
        '--exclude_path',
        type=str,
        help='Path to Excel file marking POIs to exclude',
        default=None
    )"""

    parser.add_argument(
        '--exclude',
        action="store_true",
        help='Whether to exclude certain POIs based on a predefined dictionary',
        #default=False
    )


    parser.add_argument(
        '--flip_to_right',
        action="store_true",
        help='Whether the legs should be flipped, so only right legs are used for training',
        #default=False
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
        flip_to_right=args.flip_to_right,
    )