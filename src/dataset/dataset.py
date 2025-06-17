import ast
import os

import torch
#from BIDS import NII, POI
from TPTBox import NII
from TPTBox.core.poi import POI
from torch.utils.data import Dataset

from transforms.transforms import Compose, LandMarksRandHorizontalFlip # was src.transforms.transforms
from utils.dataloading_utils import compute_surface, get_gt_pois, pad_array_to_shape


class PoiDataset(Dataset):
    def __init__(
        self,
        master_df,
        poi_indices,
        include_leg_list,
        poi_flip_pairs=None,
        input_shape=(128, 128, 96),
        transforms=None,
        flip_prob=0.5,
        include_com=False,
        poi_file_ending="poi.json",
        iterations=1,
    ):

        # If master_df has a column use_sample, filter on it
        if "use_sample" in master_df.columns:
            master_df = master_df[master_df["use_sample"]]
        self.master_df = master_df
        self.input_shape = input_shape
        self.poi_indices = poi_indices
        self.transform = Compose(transforms) if transforms else None
        if flip_prob > 0:
            self.transform = Compose(
                [self.transform, LandMarksRandHorizontalFlip(flip_prob, poi_flip_pairs)]
            )
        self.include_com = include_com
        self.poi_flip_pairs = poi_flip_pairs
        self.flip_prob = flip_prob
        self.poi_file_ending = poi_file_ending
        self.poi_idx_to_list_idx = {poi: idx for idx, poi in enumerate(poi_indices)}
        self.leg_idx_to_list_idx = {
            leg: idx for idx, bone in enumerate(include_leg_list)
        }
        self.iterations = iterations

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, index):
        data_dict = {}

        # Read the row from the master dataframe
        row = self.master_df.iloc[index]
        subject = row["subject"]
        leg = row["leg"]
        file_dir = row["file_dir"]

        # If the master_dir has a column bad_poi_list, use this to create a loss mask
        if "bad_poi_list" in self.master_df.columns:
            bad_poi_list = ast.literal_eval(row["bad_poi_list"])
            bad_poi_list = [int(poi) for poi in bad_poi_list]
            bad_poi_list = torch.tensor(bad_poi_list)
        else:
            bad_poi_list = torch.tensor([], dtype=torch.int)

        # Get the paths
        ct_path = os.path.join(file_dir, "ct.nii.gz")
        msk_path = os.path.join(file_dir, "split.nii.gz")
        subreg_path = os.path.join(file_dir, "subreg.nii.gz")
        poi_path = os.path.join(file_dir, self.poi_file_ending)

        # Load the BIDS objects
        # ct = NII.load(ct_path, seg = False)
        subreg = NII.load(subreg_path, seg=True)
        splitseg = NII.load(msk_path, seg=True)
        poi = POI.load(poi_path)

        assert (
            subreg.shape == splitseg.shape
        ), f"Subreg and splitseg shapes do not match for subject {subject}"

        zoom = (0.8, 0.8, 0.8)

        # ct.rescale_and_reorient_(
        #   axcodes_to=('L', 'A', 'S'), voxel_spacing = zoom, verbose = False
        # )
        subreg.rescale_and_reorient_(
            axcodes_to=("L", "A", "S"), voxel_spacing=zoom, verbose=False
        )
        splitseg.rescale_and_reorient_(
            axcodes_to=("L", "A", "S"), voxel_spacing=zoom, verbose=False
        )
        poi.reorient_(axcodes_to=("L", "A", "S"), verbose=False).rescale_(
            zoom, verbose=False
        )

        # Get the ground truth POIs
        poi, missing_pois = get_gt_pois(poi, leg, self.poi_indices)
        print(f"Missing POIs for subject {subject}, leg {leg}: {missing_pois}")

        poi_indices = torch.tensor(self.poi_indices)

        # Get arrays
        # ct = ct.get_array()
        subreg = subreg.get_array()
        splitseg = splitseg.get_array()

        mask = splitseg == leg

        # ct = ct * mask
        subreg = subreg * mask

        subreg, offset = pad_array_to_shape(subreg, self.input_shape)
        splitseg, _ = pad_array_to_shape(splitseg, self.input_shape)

        poi = poi + torch.tensor(offset)

        # Convert subreg and vertseg to tensors
        subreg = torch.from_numpy(subreg.astype(float))
        splitseg = torch.from_numpy(splitseg.astype(float))

        # Add channel dimension
        subreg = subreg.unsqueeze(0)
        splitseg = splitseg.unsqueeze(0)

        data_dict["input"] = subreg #splitseg
        data_dict["target"] = poi
        data_dict["target_indices"] = poi_indices

        data_dict = self.transform(data_dict) if self.transform else data_dict

        # Identify pois outside of the input shape
        max_x = self.input_shape[0] - 1
        max_y = self.input_shape[1] - 1
        max_z = self.input_shape[2] - 1

        outside_poi_indices = (
            (data_dict["target"][:, 0] < 0)
            | (data_dict["target"][:, 0] > max_x)
            | (data_dict["target"][:, 1] < 0)
            | (data_dict["target"][:, 1] > max_y)
            | (data_dict["target"][:, 2] < 0)
            | (data_dict["target"][:, 2] > max_z)
        )

        # Create a loss mask for pois shifted oustide of the image due to augmentation,
        # missing pois from the ground truth and bad pois
        loss_mask = torch.ones_like(data_dict["target"][:, 0])
        loss_mask[outside_poi_indices] = 0
        bad_poi_list_idx = [
            self.poi_idx_to_list_idx[bad_poi.item()]
            for bad_poi in bad_poi_list
            if bad_poi.item() in self.poi_indices
        ]
        loss_mask[bad_poi_list_idx] = 0
        missing_poi_list_idx = [
            self.poi_idx_to_list_idx[missing_poi.item()] for missing_poi in missing_pois
        ]
        loss_mask[missing_poi_list_idx] = 0

        data_dict["loss_mask"] = loss_mask.bool()

        transformed_mask = data_dict["input"] > 0
        surface = compute_surface(transformed_mask, iterations=self.iterations)

        data_dict["surface"] = surface
        data_dict["subject"] = str(subject)
        data_dict["leg"] = leg
        data_dict["zoom"] = torch.tensor(zoom).float()
        data_dict["offset"] = torch.tensor(offset).float()
        data_dict["ct_path"] = ct_path
        data_dict["msk_path"] = msk_path
        data_dict["subreg_path"] = subreg_path
        data_dict["poi_path"] = poi_path
        data_dict["poi_list_idx"] = torch.tensor(
            [self.poi_idx_to_list_idx[poi.item()] for poi in poi_indices]
        )
        data_dict["leg_list_idx"] = torch.tensor([self.leg_idx_to_list_idx[leg]])

        return data_dict


class LegDataset(PoiDataset):
    def __init__(
        self,
        master_df,
        input_shape=(128, 128, 96), #TODO
        transforms=None,
        flip_prob=0.5,
        include_com=False,
        include_poi_list=None,
        include_leg_list=None,
        poi_file_ending="poi.json",
        iterations=1,
    ):
        super().__init__(
            master_df,
            poi_indices=(
                include_poi_list
                if include_poi_list
                else (
                    [
                        1, 
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                        31,
                        32,
                        98,
                        99
                    ]
                    if include_com
                    else [
                        1, 
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                        31,
                        32,
                        98,
                        99
                    ]
                )
            ),
            include_vert_list=(
                include_leg_list
                if include_leg_list
                else [
                    1,
                    2
                ]
            ),
            poi_flip_pairs={
                # Middle points
                1: 1,
                2: 2,
                7: 7,
                10: 10,
                11: 11,
                12: 12,
                18: 18,
                19: 19,
                20: 20,
                23: 23,
                30: 30,
                98: 98,
                99: 99,

                # Flipped left to right
                3: 4,
                5: 6,
                8: 9,
                13: 32,
                14: 21,
                15: 22,
                16: 17,
                24: 25,
                26: 27,
                28: 29,
                31: 32,

                # Flipped right to left
                4: 3,
                6: 5,
                9: 8,
                32: 13,
                21: 14,
                22: 15,
                17: 16,
                25: 24,
                27: 26,
                29: 28,
            },
            input_shape=input_shape,
            transforms=transforms,
            flip_prob=flip_prob,
            include_com=include_com,
            poi_file_ending=poi_file_ending,
            iterations=iterations,
        )
