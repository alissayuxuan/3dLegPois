
import os
from functools import partial
from os import PathLike
from typing import Callable, Tuple

from datetime import datetime

from TPTBox import NII, BIDS_Global_info
from TPTBox.core.poi import POI
from TPTBox.core.poi_fun.poi_global import POI_Global

from elastic_deform import apply_elastic_deformation



def get_right_poi(container) -> Tuple[POI, str]:
    right_poi_query = container.new_query(flatten=True)
    right_poi_query.filter_format("poi")
    right_poi_query.filter_filetype("json")  # only nifti files
    right_poi_query.filter_self(lambda f: "RIGHT" in str(f.file["json"]).upper())
    
    if not right_poi_query.candidates:
        print("ERROR: No Right POI candidates found!")
        return None, None
    
    right_poi_candidate = right_poi_query.candidates[0]
    print(f"Loading Right POI from: {right_poi_candidate}")

    try:
        poi = POI_Global.load(right_poi_candidate.file["json"])
        return poi, str(right_poi_candidate.file["json"])
    except Exception as e:
        print(f"Error loading POI: {str(e)}")
        return None, None
    
def get_left_poi(container) -> Tuple[POI, str]:
    left_poi_query = container.new_query(flatten=True)
    left_poi_query.filter_format("poi")
    left_poi_query.filter_filetype("json") 
    left_poi_query.filter_self(lambda f: "LEFT" in str(f.file["json"]).upper())

    if not left_poi_query.candidates:
        print("ERROR: No Left POI candidates found!")
        return None, None
    
    left_poi_candidate = left_poi_query.candidates[0]
    print(f"Loading Left POI from: {left_poi_candidate}")

    try:
        poi = POI_Global.load(left_poi_candidate.file["json"])
        return poi, str(left_poi_candidate.file["json"])
    except Exception as e:
        print(f"Error loading POI: {str(e)}")
        return None, None

def get_ct(container) -> Tuple[NII, str]:
    ct_query = container.new_query(flatten=True)
    ct_query.filter_format("ct")
    ct_query.filter_filetype("nii.gz")  # only nifti files

    if not ct_query.candidates:
        print("ERROR: CT candidates found!")
        return None, None
        
    ct_candidate = ct_query.candidates[0]

    try:
        ct = ct_candidate.open_nii()
        return ct, str(ct_candidate.file["nii.gz"])
    except Exception as e:
        print(f"Error opening CT: {str(e)}")
        return None, None 

def get_splitseg(container) -> Tuple[NII, str]:
    splitseg_query = container.new_query(flatten=True)
    splitseg_query.filter_format("split")
    splitseg_query.filter_filetype("nii.gz")  # only nifti files
    #splitseg_query.filter("seg", "subreg")
    if not splitseg_query.candidates:
        print("ERROR: No splitseg candidates found!")
        return None, None
    splitseg_candidate = splitseg_query.candidates[0]

    try:
        splitseg = splitseg_candidate.open_nii()
        return splitseg, str(splitseg_candidate.file["nii.gz"])
    except Exception as e:
        print(f"Error opening splitseg: {str(e)}")
        return None, None

def get_subreg(container) -> Tuple[NII, str]:
    subreg_query = container.new_query(flatten=True)
    subreg_query.filter_format("msk")
    subreg_query.filter_filetype("nii.gz")  # only nifti files
    if not subreg_query.candidates:
        print("ERROR: No subreg candidates found!")
        return None, None

    subreg_candidate = subreg_query.candidates[0]

    try:
        subreg = subreg_candidate.open_nii()
        return subreg, str(subreg_candidate.file["nii.gz"])
    except Exception as e:
        print(f"Error opening subreg: {str(e)}")
        return None, None

def get_files(
    container,
    get_right_poi_fn: Callable,
    get_left_poi_fn: Callable,
    get_ct_fn: Callable,
    get_splitseg_fn: Callable,
    get_subreg_fn: Callable,
) -> Tuple[Tuple[POI, str], Tuple[POI, str], Tuple[NII, str], Tuple[NII, str], Tuple[NII, str]]:
    return (
        get_right_poi_fn(container),
        get_left_poi_fn(container),
        get_ct_fn(container),
        get_splitseg_fn(container),
        get_subreg_fn(container),
    )


def create_deformed_filename(original_path, new_subject, session):
    original_filename = os.path.basename(original_path)
    # Extract original subject and session from filename
    parts = original_filename.split('_')

    new_parts = []
    for part in parts:
        if part.startswith('sub-'):
            new_parts.append(new_subject)
        elif part.startswith('ses-'):
            new_parts.append(session)
        else:
            new_parts.append(part)
    
    new_filename = '_'.join(new_parts)
    

    return new_filename 
    # sub-CTFU04045_ses-20220303_sequ-204_cms_poi


def create_deformed_save_path(original_filename, new_subject, session, data_path, is_rawdata=False):
    """save deformed data in BIDS structure"""

    new_filename = create_deformed_filename(original_filename, new_subject, session)

    if is_rawdata:
        save_dir = os.path.join(data_path, "rawdata", new_subject, session)
    else:
        save_dir = os.path.join(data_path, "derivatives", new_subject, session)

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, new_filename)

    return save_path

def perform_elastic_deform(subject, container, get_data_files, data_path, control_points=10, sigma=3, deform_iterations=2):

    # Get data files and their paths
    (right_poi, right_poi_path), (left_poi, left_poi_path), (ct, ct_path), \
    (splitseg, splitseg_path), (subreg, subreg_path) = get_data_files(container)

    if any(x is None for x in [right_poi, left_poi, ct, splitseg, subreg]):
        print(f"Skipping subject {subject} - missing files")
        return

    splitseg.reorient_(("L", "A", "S"))
    subreg.reorient_(("L", "A", "S"))
    ct.reorient_(("L", "A", "S"))
    right_poi = right_poi.resample_from_to(subreg)
    left_poi = left_poi.resample_from_to(subreg)


    session = f"ses-{datetime.now().strftime('%Y%m%d')}"

    for iter in range(1, deform_iterations+1):

        splitseg_deformed, subreg_deformed, ct_deformed, right_poi_deformed, left_poi_deformed, displacement = apply_elastic_deformation(
            splitseg, 
            subreg, 
            ct, 
            right_poi, 
            left_poi,
            control_points,
            sigma
        )

        new_subject = f"sub-D{subject.replace('sub-', '')}{iter}"

        ct_deformed_path = create_deformed_save_path(ct_path, new_subject, session, data_path, is_rawdata=True)
        splitseg_deformed_path = create_deformed_save_path(splitseg_path, new_subject, session, data_path, is_rawdata=False)
        subreg_deformed_path = create_deformed_save_path(subreg_path, new_subject, session, data_path, is_rawdata=False)
        right_poi_deformed_path = create_deformed_save_path(right_poi_path, new_subject, session, data_path, is_rawdata=False)
        left_poi_deformed_path = create_deformed_save_path(left_poi_path, new_subject, session, data_path, is_rawdata=False)

        ct_deformed.save(ct_deformed_path, verbose=False)
        splitseg_deformed.save(splitseg_deformed_path, verbose=False)
        subreg_deformed.save(subreg_deformed_path, verbose=False)
        right_poi_deformed.save(right_poi_deformed_path, verbose=False)
        right_poi_deformed.to_global().save_mrk(right_poi_deformed_path)

        left_poi_deformed.save(left_poi_deformed_path, verbose=False)
        left_poi_deformed.to_global().save_mrk(left_poi_deformed_path)



if __name__ == "__main__":

    data_path = "data_preprocessing/atlas-dataset-folder-deform"


    bids_global_info = BIDS_Global_info(
        datasets=[data_path], parents=["rawdata", "derivatives"]
    )

    get_data_files = partial(
        get_files,
        get_right_poi_fn=get_right_poi,
        get_left_poi_fn=get_left_poi,
        get_ct_fn=get_ct,
        get_splitseg_fn=get_splitseg,
        get_subreg_fn=get_subreg,
    )

    for subject, container in bids_global_info.enumerate_subjects():
        perform_elastic_deform(subject, container, get_data_files, data_path, 
                               control_points=10, sigma=3, deform_iterations=2)
        



