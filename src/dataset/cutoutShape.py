import os
import numpy as np
from TPTBox import NII
from TPTBox.core.poi import POI

import pandas as pd


def find_max_shape():
    """
    finds the maximum shape of the subregs in the cutout folder.
    """
    base_dir = 'data_preprocessing/cutout-folder/cutouts-folder-deform_more/fov_cut/patella' # change!

    max_shape = None

    for ws_folder in os.listdir(base_dir):
        ws_path = os.path.join(base_dir, ws_folder)
        if not os.path.isdir(ws_path):
            continue
        for subfolder in os.listdir(ws_path):
            sub_path = os.path.join(ws_path, subfolder)
            if not os.path.isdir(sub_path):
                continue
            subreg_path = os.path.join(sub_path, 'subreg.nii.gz')
            split_path = os.path.join(sub_path, 'split.nii.gz')
            if os.path.isfile(subreg_path):
                seg_mask = NII.load(subreg_path, seg=True) 
                shape = seg_mask.shape  # tuple mit Dimensionen
                #print(shape)
                if max_shape is None:
                    max_shape = shape
                else:
                    # max_shape wird komponentenweise maximiert
                    max_shape = tuple(max(m, s) for m, s in zip(max_shape, shape))
            if os.path.isfile(split_path):
                split_mask = NII.load(split_path, seg=True) 
                split_shape = split_mask.shape  # tuple mit Dimensionen
                if split_shape != shape:
                    print(f"subreg path: {subreg_path}")
                    print(f"SHAPE MISMATCH - subreg: {shape}, split {split_shape}\n\n")
                          
                

    print("Maximale Shape aller subreg.nii.gz Dateien:", max_shape)

def get_all_pois():

    base_dir = 'data_preprocessing/cutouts-folder/cutouts_exclude'  # change!

    unique_pois = set()

    for ws_folder in os.listdir(base_dir):
        ws_path = os.path.join(base_dir, ws_folder)
        if not os.path.isdir(ws_path):
            continue
        for subfolder in os.listdir(ws_path):
            sub_path = os.path.join(ws_path, subfolder)
            if not os.path.isdir(sub_path):
                continue
            file_path = os.path.join(sub_path, 'poi.json')
            if os.path.isfile(file_path):
                poi = POI.load(file_path) 
                poi_centroids = poi.centroids
                for poi_key in poi_centroids:
                    unique_pois.add(poi_key)
    
    return sorted(unique_pois, key=lambda x: (x[0], x[1]))
                
def rescale_cutouts(zoom:tuple):
    """
    rescales all files to specified zoom
    """
    base_dir = 'data_preprocessing/cutout-folder/cutouts-registration/patella' # change!
    save_dir = 'data_preprocessing/cutout-folder/cutouts-registration-1.5_zoom/patella' # change!


    for ws_folder in os.listdir(base_dir):
        ws_path = os.path.join(base_dir, ws_folder)
        if not os.path.isdir(ws_path):
            continue
        for subfolder in os.listdir(ws_path):
            sub_path = os.path.join(ws_path, subfolder)
            if not os.path.isdir(sub_path):
                continue

            output_dir = os.path.join(save_dir, ws_folder, subfolder)
            os.makedirs(output_dir, exist_ok=True)

            files_to_process = [
                ('subreg.nii.gz', True), #(filename, is_segmentation)
                ('boneseg.nii.gz', True),
                ('split.nii.gz', True),
            ]
            # Segmentation masks and CT scans
            for filename, is_seg in files_to_process:
                input_path = os.path.join(sub_path, filename)
                output_path = os.path.join(output_dir, filename)

                if os.path.isfile(input_path):
                    nii_data = NII.load(input_path, seg=is_seg) 
                    nii_data.rescale_(zoom)
                    nii_data.save(output_path, verbose=False)

            # POI files
            poi_path = os.path.join(sub_path, 'poi.json')
            if os.path.isfile(poi_path):
                save_poi_path = os.path.join(output_dir, 'poi.json')
                save_global_poi_path = os.path.join(output_dir, 'poi_global.json')
                
                poi = POI.load(poi_path) 
                poi.rescale_(zoom)
                poi.save(save_poi_path, verbose=False)
                poi.to_global().save_mrk(save_global_poi_path)           

def copy_and_update_master_df(base_dir: str, save_dir: str, old_folder: str = "cutouts-registration", new_folder: str = "cutouts-registration-1.5_zoom"):
    """
    Copies master_df.csv from base_dir to save_dir and updates file_dir paths
    
    Args:
        base_dir: Source directory containing master_df.csv
        save_dir: Destination directory for the updated master_df.csv
        old_folder: Folder name to replace in file_dir (default: "cutouts")
        new_folder: New folder name (default: "cutouts-0.5_zoom")
    """
    # Load the master_df
    master_df_path = os.path.join(base_dir, 'master_df.csv')
    df = pd.read_csv(master_df_path)
    
    # Update the file_dir column
    df['file_dir'] = df['file_dir'].str.replace(f'/{old_folder}/', f'/{new_folder}/', regex=False)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the updated dataframe
    output_path = os.path.join(save_dir, 'master_df.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Updated master_df.csv saved to: {output_path}")
    print(f"Updated {len(df)} rows, replacing '{old_folder}' with '{new_folder}' in file paths")


if __name__ == "__main__":
    find_max_shape()
    #print(get_all_pois())
    #rescale_cutouts((1.5, 1.5, 1.5))

    #copy_and_update_master_df(
    #    base_dir='data_preprocessing/cutout-folder/cutouts-registration/patella',
    #    save_dir='data_preprocessing/cutout-folder/cutouts-registration-1.5_zoom/patella'
    #)