import elasticdeform
import numpy as np
import os
from scipy.ndimage import map_coordinates
from TPTBox import NII
from TPTBox.core.poi import POI


def generate_displacement(shape, control_points=4, sigma=5):
    """
    Generates a random displacement field for elastic deformation. It will define 
    how each point is shifted. Sigma defines how strong the shifts are.
    """
    return np.random.randn(3, control_points, control_points, control_points) * sigma

def interpolate_displacement_to_grid(displacement, target_shape):
    """
    Interpolates the displacement field to the target shape: Interpolation between the 
    control points of the displacement field. Now each point has a "shifting vector".
    """
    X, Y, Z = target_shape 
    control_points = displacement.shape[1]  

    # create coordinate grid for interpolation
    coords_z = np.linspace(0, control_points-1, Z) # creates Z values between 0 and control_points-1
    coords_y = np.linspace(0, control_points-1, Y) 
    coords_x = np.linspace(0, control_points-1, X)
    
    # create 3D meshgrid 
    zz_interp, yy_interp, xx_interp = np.meshgrid(
        coords_z, coords_y, coords_x, indexing="ij"
    )
    
    displacement_resampled = np.zeros((3, Z, Y, X))
    
    # Interpolation
    for axis in range(3):
        displacement_resampled[axis] = map_coordinates(
            displacement[axis], 
            [zz_interp, yy_interp, xx_interp],  
            order=3,  
            mode="constant", #"reflect",
            prefilter=False  
        )
    
    return displacement_resampled


def elastic_deform_pois(poi, seg_shape, displacement):
    """
    Deforms POIs with cubic spline interpolation of the displacement fields
    
    Args:
        poi: POI object
        seg_shape: segmentation shape (X, Y, Z) 
        displacement: displacement field (3, control_points, control_points, control_points)
    
    Returns:
        poi: deformed POIs 
    """
    
    X, Y, Z = seg_shape  
    
    displacement_resampled = interpolate_displacement_to_grid(displacement, seg_shape)
    
    for label in poi.keys():
        x, y, z = poi.centroids[label] 
        
        # check if POI is within bounds
        if not (0 <= x < X and 0 <= y < Y and 0 <= z < Z):
            print(f"Warning: POI {label} at ({x:.2f}, {y:.2f}, {z:.2f}) out of bounds")
            continue
            
        coords = [[z], [y], [x]] 
        
        # interpolate between each "pixel"
        dx = map_coordinates(displacement_resampled[2], coords, 
                           order=3, mode='constant', prefilter=False)[0]  # x-displacement
        dy = map_coordinates(displacement_resampled[1], coords, 
                           order=3, mode='constant', prefilter=False)[0]  # y-displacement
        dz = map_coordinates(displacement_resampled[0], coords, 
                           order=3, mode='constant', prefilter=False)[0]  # z-displacement
        
        # new position
        new_x = x + dx
        new_y = y + dy
        new_z = z + dz
        
        poi.centroids[label] = (new_x, new_y, new_z)
        print(f"POI {label}: ({x:.2f}, {y:.2f}, {z:.2f}) -> ({new_x:.2f}, {new_y:.2f}, {new_z:.2f})")

    return poi
    
def elastic_deform_seg(seg, displacement):
    """
    Performs elastic deformation on segmentation mask
    
    Args:
        seg: segmentation mask
        displacement: displacement field (3, control_points, control_points, control_points)
    
    Returns:
        seg: deformed segmentation mask
    """
    seg_array = seg.get_array()
    
    seg_deformed = elasticdeform.deform_grid(
        seg_array,
        displacement,
        axis=(0, 1, 2),  # deform all 3 spatial dimensions
        order=0,  # nearest neighbor for labels (no new labels)
        mode='constant'#'reflect'  # boundry handling
    )
    
    seg.set_array_(seg_deformed)
    return seg

def elastic_deform_ct(ct, displacement):
    """
    Performs elastic deformation on ct scan
    
    Args:
        ct: ct scan mask
        displacement: displacement field (3, control_points, control_points, control_points)
    
    Returns:
        ct: deformed ct scan
    """
    
    ct_array = ct.get_array()
    
    ct_deformed = elasticdeform.deform_grid(
        ct_array,
        displacement,
        axis=(0, 1, 2),  # deform all 3 spatial dimensions
        order=3,  # cubic spline interpolation
        mode='constant'# boundry handling
    )
    
    ct.set_array_(ct_deformed)
    return ct
    
def apply_elastic_deformation(splitseg, subreg, ct, right_poi, left_poi,  
                            control_points=10, sigma=3, save_dir=None, displacement=None):
    """
    applies elastic deformation, (optionally) saves, and  returns result
    """
    # Get shape
    splitseg_shape = splitseg.shape
        
    # generate Displacement-Field
    if displacement is None:
        displacement = generate_displacement(splitseg_shape, control_points, sigma)
        print(f"Generated displacement field with shape: {displacement.shape}")
    
    splitseg_deformed = elastic_deform_seg(splitseg, displacement)
    subreg_deformed = elastic_deform_seg(subreg, displacement)
    ct_deformed = elastic_deform_ct(ct, displacement)

    right_poi_deformed = elastic_deform_pois(right_poi, splitseg_shape, displacement)
    left_poi_deformed = elastic_deform_pois(left_poi, splitseg_shape, displacement)
    
    if save_dir:
        splitseg_path = os.path.join(save_dir, "split_deform.nii.gz")
        subreg_path = os.path.join(save_dir, "subreg_deform.nii.gz")
        ct_path = os.path.join(save_dir, "ct_deform.nii.gz")
        right_poi_path = os.path.join(save_dir, "right_poi_deform.json")
        right_poi_global_path = os.path.join(save_dir, "poi_deform_global.json")
        left_poi_path = os.path.join(save_dir, "left_poi_deform.json")
        left_poi_global_path = os.path.join(save_dir, "left_poi_deform_global.json")

        splitseg_deformed.save(splitseg_path)
        print(f"Saved deformed segmentation to: {splitseg_path}")
        subreg_deformed.save(subreg_path)
        print(f"Saved deformed subregion to: {subreg_path}")

        ct_deformed.save(ct_path)
        print(f"Saved deformed CT to: {ct_path}")
        
        right_poi_deformed.save(right_poi_path)
        right_poi_deformed.to_global().save_mrk(right_poi_global_path)
        print(f"Saved deformed right POIs to: {right_poi_path}")

        left_poi_deformed.save(right_poi_path)
        left_poi_deformed.to_global().save_mrk(left_poi_global_path)
        print(f"Saved deformed left POIs to: {left_poi_path}")
    
    return splitseg_deformed, subreg_deformed, ct_deformed, right_poi_deformed, left_poi_deformed, displacement

if __name__ == "__main__":

    splitseg_path = "data_preprocessing/cutouts-folder_test/side_cut/CTFU00099/left/split.nii.gz"
    subreg_path = "data_preprocessing/cutouts-folder_test/side_cut/CTFU00099/left/subreg.nii.gz"

    poi_path = "data_preprocessing/cutouts-folder_test/side_cut/CTFU00099/left/poi.json"
    #save_dir = "data_preprocessing/deform/elastic_deform_coords/order3/CTFU00099/left/"

    fine_control_configs = [
        (6, 4), 
        (8, 3), 
        (8, 4), 
        (10, 3), 
    ]

    for cp, sigma in fine_control_configs:
        print(f"Testing control_points={cp}, sigma={sigma}")
        save_dir = f"data_preprocessing/deform/fine_control_configs/pixel-order3/test_cp{cp}_sigma{sigma}"
        try: 
            splitseg_deformed, subreg_deformed, poi_deformed, displacement = apply_elastic_deformation(
                splitseg_path=splitseg_path,
                subreg_path=subreg_path,
                poi_path=poi_path,
                poi_path=poi_path,
                save_dir=save_dir,
                control_points=cp,  # Mehr Kontrollpunkte für feinere Deformation
                sigma=sigma,  # Deformationsstärke
                displacement=None
            )       
            
            print("Elastic deformation completed successfully!\n\n")
            
        except Exception as e:
            print(f"Error: {e}")