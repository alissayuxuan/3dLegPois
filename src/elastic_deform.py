import elasticdeform
import numpy as np
from scipy.ndimage import map_coordinates
from TPTBox import NII
from TPTBox.core.poi import POI


def generate_displacement(shape, control_points=4, sigma=5):
    """
    Generiert ein zufälliges Displacement-Feld für elastische Deformation.
    
    Args:
        shape: Shape des Zielbildes (D, H, W)
        control_points: Anzahl der Kontrollpunkte pro Dimension
        sigma: Standardabweichung für die zufällige Deformation
    
    Returns:
        displacement: Array mit Shape (3, control_points, control_points, control_points)
    """
    return np.random.randn(3, control_points, control_points, control_points) * sigma


def interpolate_displacement_to_grid_old(displacement, target_shape):
    """
    Interpoliert das Displacement-Feld auf die Zielgröße des Bildes.
    
    Args:
        displacement: (3, control_points, control_points, control_points)
        target_shape: Zielgröße (X, Y, Z) - TPTBox Format
    
    Returns:
        displacement_resampled: (3, Z, Y, X) - in Array-Zugriff Format
    """
    X, Y, Z = target_shape  # TPTBox Format
    control_points = displacement.shape[1]  # Annahme: kubisches Gitter
    
    # Erstelle Koordinatengitter für das Zielbild
    # Interpolation für jede Dimension separat
    xx, yy, zz = np.meshgrid(
        np.linspace(0, control_points-1, X),
        np.linspace(0, control_points-1, Y), 
        np.linspace(0, control_points-1, Z),
        indexing="ij"
    )
    # Stack in der Reihenfolge für Array-Zugriff: [z, y, x]
    grid_coords = np.stack([zz, yy, xx])  # (3, Z, Y, X)
    
    # Interpoliere jede Dimension des Displacement-Feldes
    displacement_resampled = np.stack([
        map_coordinates(displacement[d], grid_coords, order=3, mode="reflect")
        for d in range(3)
    ])  # shape: (3, Z, Y, X)
    
    return displacement_resampled


def interpolate_displacement_to_grid(displacement, target_shape):
    """
    Interpoliert das Displacement-Feld auf die Zielgröße des Bildes.
    
    Args:
        displacement: (3, control_points, control_points, control_points)
        target_shape: Zielgröße (X, Y, Z) - TPTBox Format
    
    Returns:
        displacement_resampled: (3, Z, Y, X) - in Array-Zugriff Format
    """
    X, Y, Z = target_shape  # TPTBox Format
    control_points = displacement.shape[1]  # Annahme: kubisches Gitter
    
    # KRITISCH: Verwende dieselbe Interpolationsmethode wie elasticdeform
    # elasticdeform verwendet standardmäßig lineare Interpolation zwischen Kontrollpunkten
    
    # Erstelle Koordinatengitter für die Interpolation
    # Für jede Zieldimension: von 0 bis (control_points-1)
    coords_z = np.linspace(0, control_points-1, Z)
    coords_y = np.linspace(0, control_points-1, Y) 
    coords_x = np.linspace(0, control_points-1, X)
    
    # Erstelle 3D-Gitter für die Interpolation
    zz_interp, yy_interp, xx_interp = np.meshgrid(
        coords_z, coords_y, coords_x, indexing="ij"
    )
    
    # Interpoliere jede Dimension des Displacement-Feldes separat
    displacement_resampled = np.zeros((3, Z, Y, X))
    
    # Wichtig: Die Reihenfolge muss mit elasticdeform übereinstimmen
    # elasticdeform erwartet displacement in der Form (axis, control_points...)
    for axis in range(3):
        displacement_resampled[axis] = map_coordinates(
            displacement[axis], 
            [zz_interp, yy_interp, xx_interp],  # Koordinaten für Interpolation
            order=1,  # Lineare Interpolation (wie elasticdeform)
            mode="reflect",
            prefilter=False  # Same as elasticdeform default
        )
    
    return displacement_resampled



def deform_poi(poi_coords, deformed_grid):
    """
    Deformiert einen einzelnen POI basierend auf dem Deformationsgitter.
    
    Args:
        poi_coords: Tuple (x, y, z) - Koordinaten des POI (TPTBox Format)
        deformed_grid: (3, Z, Y, X) - enthält die neuen Positionen für jeden Voxel
    
    Returns:
        deformed_poi: Tuple (x, y, z) mit den neuen Koordinaten
    """
    x, y, z = poi_coords  # POI ist in (x, y, z) Format
    Z, Y, X = deformed_grid.shape[1:]  # Array-Dimensionen
    
    # Konvertiere zu Array-Indizes: (x,y,z) -> (z,y,x) für Array-Zugriff
    # Prüfe ob POI innerhalb der Bildgrenzen liegt
    if not (0 <= x < X and 0 <= y < Y and 0 <= z < Z):
        print(f"Warning: POI ({x}, {y}, {z}) außerhalb der Bildgrenzen (X={X}, Y={Y}, Z={Z})")
        return poi_coords  # Gib ursprüngliche Koordinaten zurück
    
    # Interpoliere die neuen Koordinaten an der POI-Position
    # Wichtig: Array-Zugriff ist [z, y, x], aber POI ist (x, y, z)
    deformed_coords = [
        map_coordinates(deformed_grid[d], [[z], [y], [x]], order=1, mode='nearest')[0]
        for d in range(3)
    ]
    
    # Rückgabe in (x, y, z) Format - konvertiere von Array-Koordinaten zurück
    # deformed_coords[0] = neue z-Koordinate
    # deformed_coords[1] = neue y-Koordinate  
    # deformed_coords[2] = neue x-Koordinate
    return (deformed_coords[2], deformed_coords[1], deformed_coords[0])  # (x, y, z)


def elastic_deform_pois(poi, seg_shape, displacement):
    """
    Deformiert POIs basierend auf dem gleichen Displacement-Feld.
    
    Args:
        poi: POI object
        seg_shape: Shape der Segmentierung (X, Y, Z) - TPTBox Format
        displacement: Displacement-Feld (3, control_points, control_points, control_points)
    
    Returns:
        deformed_centroids: Dict mit deformierten POI-Koordinaten {label: (x, y, z)}
    """
    X, Y, Z = seg_shape  # TPTBox: (X, Y, Z) entspricht (Width, Height, Depth)
    
    # Interpoliere das Displacement auf die Bildgröße
    displacement_resampled = interpolate_displacement_to_grid(displacement, seg_shape)

    xx, yy, zz = np.meshgrid(
        np.arange(X), np.arange(Y), np.arange(Z), indexing="ij"
    )
    # Stack in Array-Zugriff Reihenfolge: [z, y, x] für numpy arrays
    original_grid = np.stack([zz, yy, xx])  # (3, Z, Y, X) - Format: [z, y, x] für Array-Zugriff
    
    # Berechne die neuen Koordinaten (ursprüngliche Position + Displacement)
    deformed_grid = original_grid + displacement_resampled  # shape: (3, Z, Y, X)
    
    # deform every poi
    for label in poi.keys():
        poi_coord = poi.centroids[label]
        try:
            deformed = deform_poi(poi_coord, deformed_grid)
            poi.centroids[label] = deformed
        except Exception as e:
            print(f"Error deforming POI {label} at {poi_coord}: {e}")
             
    return poi

def elastic_deform_pois(poi, seg_shape, displacement):
    """
    Deformiert POIs basierend auf dem gleichen Displacement-Feld.
    
    Args:
        centroids: Dict mit {label: (x, y, z)} - POI-Koordinaten im TPTBox Format
        seg_shape: Shape der Segmentierung (X, Y, Z) - TPTBox Format
        displacement: Displacement-Feld (3, control_points, control_points, control_points)
    
    Returns:
        deformed_centroids: Dict mit deformierten POI-Koordinaten {label: (x, y, z)}
    """
    X, Y, Z = seg_shape  # TPTBox Format
    
    displacement_resampled = interpolate_displacement_to_grid(displacement, seg_shape)
    
    for label in poi.keys():
        x, y, z = poi.centroids[label]  # TPTBox Format: (x, y, z)
        
        # Hole das Displacement an der POI-Position
        # displacement_resampled hat Format (3, Z, Y, X) = (axis, z, y, x)
        dx = displacement_resampled[2, int(z), int(y), int(x)]  # x-displacement 
        dy = displacement_resampled[1, int(z), int(y), int(x)]  # y-displacement
        dz = displacement_resampled[0, int(z), int(y), int(x)]  # z-displacement
        
        # Neue Position = alte Position + displacement
        new_x = x + dx
        new_y = y + dy  
        new_z = z + dz
        
        poi.centroids[label] = (new_x, new_y, new_z)
        print(f"POI {label}: ({x:.1f}, {y:.1f}, {z:.1f}) -> ({new_x:.1f}, {new_y:.1f}, {new_z:.1f})")

    return poi

def elastic_deform_seg(seg, displacement):
    """
    Deformiert eine Segmentationsmaske elastisch.
    
    Args:
        seg: NII object mit Segmentationsdaten
        displacement: Displacement-Feld (3, control_points, control_points, control_points)
    
    Returns:
        seg: Deformierte Segmentationsmaske (in-place Modifikation)
    """
    seg_array = seg.get_array()
    
    seg_deformed = elasticdeform.deform_grid(
        seg_array,
        displacement,
        axis=(0, 1, 2),  # deform all 3 spatial dimensions
        order=0,  # nearest neighbor for labels (no new labels)
        mode='reflect'  # boundry handling
    )
    
    seg.set_array_(seg_deformed)
    return seg

def apply_elastic_deformation(seg_path, poi_path, output_seg_path, output_poi_path, 
                            control_points=4, sigma=8):
    """
    Hauptfunktion: Lädt Seg/POI, wendet elastische Deformation an und speichert Ergebnisse.
    
    Args:
        seg_path: Pfad zur Segmentationsdatei
        poi_path: Pfad zur POI-Datei  
        output_seg_path: Ausgabepfad für deformierte Segmentation
        output_poi_path: Ausgabepfad für deformierte POIs
        control_points: Anzahl Kontrollpunkte pro Dimension
        sigma: Deformationsstärke
    """
    # Lade Segmentation
    seg = NII.load(seg_path, seg=True)
    seg_shape = seg.shape
    print(f"Segmentation shape: {seg_shape}")
    
    # Lade POIs
    poi = POI.load(poi_path)
    # Extrahiere centroids Dictionary (bereits im richtigen Format)
    
    # Generiere Displacement-Feld
    displacement = generate_displacement(seg_shape, control_points, sigma)
    print(f"Generated displacement field with shape: {displacement.shape}")
    
    seg_deformed = elastic_deform_seg(seg, displacement)
    poi_deformed = elastic_deform_pois(poi, seg_shape, displacement)
    
    seg_deformed.save(output_seg_path)
    print(f"Saved deformed segmentation to: {output_seg_path}")

    poi_deformed.save(output_poi_path)
    new_filename = output_poi_path.replace("_poi.json", "_poi_global.json")
    poi_deformed.to_global().save_mrk(new_filename)
    print(f"Saved deformed POIs to: {output_poi_path}")
    
    return seg_deformed, poi_deformed, displacement


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Beispielaufruf
    seg_path = "dataset/data_preprocessing/cutouts-folder/cutouts_all/femur/CTFU00099/1/split.nii.gz"
    poi_path = "dataset/data_preprocessing/cutouts-folder/cutouts_all/femur/CTFU00099/1/poi.json"
    output_seg_path = "dataset/data_preprocessing/deform/deform_seg.nii.gz"
    output_poi_path = "dataset/data_preprocessing/deform/deform_poi.json"

    #seg = NII.load(seg_path, seg=True)
    #poi = POI.load(poi_path)

    #print(f"poi-centroids: {poi.centroids}")
    #print(f"poi-keys: {poi.keys()}")
    #print(f"poi-coords: {poi.values()}")
    #print(f"poi 1 : {poi.centroids[(1, 1)]}")
    #poi.centroids[(1, 1)] = (10, 10, 10)  # Beispieländerung
    #print(f"poi 1 after change: {poi.centroids[(1, 1)]}")


    #seg_shape = seg.shape
    #displacement = generate_displacement(seg_shape, 4, 8)
    #print(f"Generated displacement field with shape: {displacement.shape}")
 
    try:
        seg_deformed, poi_deformed, displacement = apply_elastic_deformation(
            seg_path=seg_path,
            poi_path=poi_path,
            output_seg_path=output_seg_path,
            output_poi_path=output_poi_path,
            control_points=4,  # Mehr Kontrollpunkte für feinere Deformation
            sigma=8  # Deformationsstärke
        )
        
        print("Elastic deformation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        


"""import elasticdeform
import numpy as np
from scipy.ndimage import map_coordinates
from TPTBox import NII
from TPTBox.core.poi import POI


def generate_displacement(control_points=4, sigma=5):
    return np.random.randn(3, control_points, control_points, control_points) * sigma

def deform_poi(poi, deformed_grid):
    # poi: Tuple (z, y, x)
    # deformed_grid: (3, D, H, W) – enthält neue Positionen
    z, y, x = poi
    deformed_poi = [
        map_coordinates(deformed_grid[d], [[z], [y], [x]], order=3, mode='nearest')[0]
        for d in range(3)
    ]
    return tuple(deformed_poi)


def elastic_deform_seg(seg, displacement):
    # deform the segmentation
    seg_array = seg.get_array()
    seg_deformed = elasticdeform.deform_grid(
        seg_array,
        displacement,
        axis=(0, 1, 2), # ???? wofür?
        order=0  # ???? wichtig für Labels?
    )
    seg.set_array(seg_deformed)
    return seg

def elastic_deform_pois(pois, seg, displacement):

    D, H, W = seg.shape
    # Koordinatengitter (shape = (3, D, H, W))
    zz, yy, xx = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing="ij"
    )
    grid = np.stack([zz, yy, xx])  # (3, D, H, W)

    # Interpoliere das Displacement auf das Gitter (auf gleiche Auflösung wie Bild)
    displacement_resampled = np.stack([
        map_coordinates(displacement[d], grid, order=3, mode="nearest")
        for d in range(3)
    ])  # shape: (3, D, H, W)

    # Neue Koordinaten
    deformed_grid = grid + displacement_resampled  # shape: (3, D, H, W)



    deformed_pois = [deform_poi(p, deformed_grid) for p in pois]
"""
    





