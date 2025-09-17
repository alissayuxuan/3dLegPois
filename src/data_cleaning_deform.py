# As the elastic deform is not perfectly accurate and some POIs don't end up on the 
# surface as they should. We either filter those POIs out or project them to the surface
# NOTE: not all POIs should be on the surface
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import surface_project_coords
from utils.dataloading_utils import compute_surface

import torch
import ast

import pandas as pd
from copy import deepcopy
import shutil

from TPTBox import NII
from TPTBox.core.poi import POI


SURFACE_POIS = [
    8, 9, 10, 13, 14, 15, 16, 17, 18, 21, 22, # femur
    1, 2, 3, 4, 5, 6, 7, # patella
    114, 121, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 98, 99, # lowerleg
]
  

def project_selected_pois(poi_obj, surface, surface_pois):
    # get coordinates of the pois_to_proj: [(leg_id, poi_id), ...]

    coords_list = [] # stores coordinates
    poi_keys = [] # stores correspondinng (leg_id, poi_id) keys
    for leg_id, poi_id in surface_pois:
            if (leg_id, poi_id) in poi_obj:
                coords_list.append(poi_obj[leg_id, poi_id])
                poi_keys.append((leg_id, poi_id))
    
    if not coords_list:
         return poi_obj

    coords_tensor = torch.tensor(coords_list, dtype=torch.float32)

    print(f"coords_tensor: {coords_tensor}")

    proj_coords, _ = surface_project_coords(coords_tensor, surface)

    print(f"proj_coords: {proj_coords}")
    proj_coords_list = proj_coords.tolist()


    if len(proj_coords.shape) == 3 and proj_coords.shape[0] == 1:
        # Remove the batch dimension: [1, N, 3] -> [N, 3]
        proj_coords_list = proj_coords[0].tolist()


    print(f"proj_coords_list: {proj_coords_list}")

    for i in range(len(proj_coords)):
         leg_id, poi_id = poi_keys[i]
         print(f"POI before projection: {poi_obj.centroids}")
         poi_obj[leg_id, poi_id] = proj_coords_list[i]
         print(f"Projected POI ({leg_id}, {poi_id}) to {proj_coords_list[i]}")
         print(f"POI after projection: {poi_obj.centroids}")

    return poi_obj



def filter_distant_pois(poi_obj, surface, surface_pois, distance_threshold=2.0):
    """Find all poi_ids of surface_pois in the poi_obj that are too far away from the surface."""
    # return the list of those poi_ids
    surface_points = surface.nonzero(as_tuple=False).float()
    
    if len(surface_points) == 0:
        print("Warning: No surface points found")
        return []
    
    distant_poi_ids = []
    
    for leg_id, poi_id in surface_pois:
        if (leg_id, poi_id) not in poi_obj:
            continue
            
        # Get POI coordinates
        poi_coord = poi_obj[leg_id, poi_id]
        poi_tensor = torch.tensor(poi_coord, dtype=torch.float32)
        
        # Calculate distances to all surface points
        distances = torch.sqrt(((surface_points - poi_tensor) ** 2).sum(dim=1))
        min_distance = torch.min(distances).item()
        
        # Collect POIs that are too far away from the surface
        if min_distance > distance_threshold:
            distant_poi_ids.append(poi_id)
            print(f"POI ({leg_id}, {poi_id}) is {min_distance:.2f} voxels from surface -> bad!")
    
    return distant_poi_ids


def clean_surface_pois(master_df, base_save_dir):

    proj_entries = []
    outliers_entries = []
    region = master_df["region"].iloc[0]

    for index in range(len(master_df)):
        row = master_df.iloc[index]
        subject = row["subject"]
        leg = row["leg"]
        file_dir = row["file_dir"]

        if "bad_poi_list" in master_df.columns:
            bad_poi_list = ast.literal_eval(row["bad_poi_list"])
            bad_poi_list = [int(poi) for poi in bad_poi_list]
        else:
            bad_poi_list = torch.tensor([], dtype=torch.int)

        ct_path = os.path.join(file_dir, "ct.nii.gz")
        msk_path = os.path.join(file_dir, "split.nii.gz")
        subreg_path = os.path.join(file_dir, "subreg.nii.gz")
        poi_path = os.path.join(file_dir, "poi.json")
        cutout_slice_path = os.path.join(file_dir, "cutout_slice_indices.json")

        ct = NII.load(ct_path, seg=False)
        subreg = NII.load(subreg_path, seg=True)
        splitseg = NII.load(msk_path, seg=True)
        poi = POI.load(poi_path)

        # Create Surface pois for specific leg
        leg_id = int(leg) if isinstance(leg, str) else leg
        surface_pois = [(leg_id, poi_id) for poi_id in SURFACE_POIS]

        # Compute surface
        surface = compute_surface(splitseg, iterations=1)

        # Project POIs to surface and get bad_poi_list (surface pois that are too far away from surface)
        projected_poi = deepcopy(poi)
        print("projecting surface pois")
        projected_poi = project_selected_pois(projected_poi, surface, surface_pois)
        print(f"projected_poi: {projected_poi.centroids}")
        
        print("filter distant pois")
        bad_poi_list_surface = filter_distant_pois(poi, surface, surface_pois)

        # Save to new directory
        proj_path = os.path.join(base_save_dir, "cutouts-proj_to_surface", region, subject, str(leg))
        os.makedirs(proj_path, exist_ok=True)

        proj_poi_path = os.path.join(proj_path, "poi.json")
        proj_poi_global_path = os.path.join(proj_path, "poi_global.mrk.json")
        proj_ct_path = os.path.join(proj_path, "ct.nii.gz")
        proj_subreg_path = os.path.join(proj_path, "subreg.nii.gz")
        proj_split_path = os.path.join(proj_path, "split.nii.gz")
        proj_cutout_slice_path = os.path.join(proj_path, "cutout_slice_indices.json")

        
        outliers_path = os.path.join(base_save_dir, "cutouts-removed_not_on_surface", region, subject, str(leg))
        os.makedirs(outliers_path, exist_ok=True)

        outliers_poi_path = os.path.join(outliers_path, "poi.json")
        outliers_poi_global_path = os.path.join(outliers_path, "poi_global.mrk.json")
        outliers_ct_path = os.path.join(outliers_path, "ct.nii.gz")
        outliers_subreg_path = os.path.join(outliers_path, "subreg.nii.gz")
        outliers_split_path = os.path.join(outliers_path, "split.nii.gz")
        outliers_cutout_slice_path = os.path.join(outliers_path, "cutout_slice_indices.json")
        

        # Create 2 different cutout folders. Save the files to the new one
        projected_poi.save(proj_poi_path)
        projected_poi.to_global().save_mrk(proj_poi_global_path)
        ct.save(proj_ct_path)
        subreg.save(proj_subreg_path)
        splitseg.save(proj_split_path)

        proj_entries.append({
            "region": region,
            "subject": subject,
            "leg": leg,
            "file_dir": proj_path,
            "bad_poi_list": bad_poi_list
        })

        
        poi.save(outliers_poi_path)
        poi.to_global().save_mrk(outliers_poi_global_path)
        ct.save(outliers_ct_path)
        subreg.save(outliers_subreg_path)
        splitseg.save(outliers_split_path)

        updated_bad_poi_list = list(set(bad_poi_list + bad_poi_list_surface))
      

        outliers_entries.append({
            "region": region,
            "subject": subject,
            "leg": leg,
            "file_dir": outliers_path,
            "bad_poi_list": updated_bad_poi_list
        })
        
        if os.path.exists(cutout_slice_path):
            shutil.copy2(cutout_slice_path, proj_cutout_slice_path)
            shutil.copy2(cutout_slice_path, outliers_cutout_slice_path)
        
    proj_master_df = pd.DataFrame(proj_entries)
    proj_region_folder = os.path.join(base_save_dir, "cutouts-proj_to_surface", region)
    os.makedirs(proj_region_folder, exist_ok=True)
    proj_master_df.to_csv(os.path.join(proj_region_folder, "master_df.csv"), index=False)

    
    outliers_master_df = pd.DataFrame(outliers_entries)
    outliers_region_folder = os.path.join(base_save_dir, "cutouts-removed_not_on_surface", region)
    os.makedirs(outliers_region_folder, exist_ok=True)
    outliers_master_df.to_csv(os.path.join(outliers_region_folder, "master_df.csv"), index=False)
    

if __name__ == "__main__":
    master_df = pd.read_csv("dataset/data_preprocessing/cutout-folder/cutouts-folder-deform_more/fov_cut/lowerleg/master_df.csv")
    base_save_dir = "dataset/data_preprocessing/cutout-folder"

    clean_surface_pois(master_df, base_save_dir)
     
