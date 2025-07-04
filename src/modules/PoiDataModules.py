import os
from functools import partial
from os import PathLike
from typing import TypeVar

import pandas as pd
import pytorch_lightning as pl
import torch
#from BIDS import BIDS_Global_info
from TPTBox import BIDS_Global_info


from pqdm.processes import pqdm
from torch.utils.data import DataLoader

from dataset.dataset import FemurDataset, PatellaDataset, LowerLegDataset, PoiDataset
from transforms.transforms import create_transform
"""
from utils.dataloading_utils import (
    get_ct,
    get_files,
    get_gruber_poi,
    get_subreg,
    get_vertseg,
    process_container,
)
"""

PoiType = TypeVar("PoiType", bound=PoiDataset)


class POIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        master_df: PathLike,
        train_subjects: list,
        val_subjects: list,
        test_subjects: list,
        input_shape: tuple = (190, 215, 1165),
        flip_prob: float = 0.5,
        transform_config: dict | None = None,
        include_com: bool = False,
        include_poi_list=None,
        include_leg_list=None,
        batch_size: int = 1,
        num_workers: int = 0,
        poi_file_ending: str = "poi.json",
        surface_erosion_iterations: int = 1,
    ):
        super().__init__()
        self.dataset = dataset
        self.master_df_path = master_df
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.test_subjects = test_subjects
        self.input_shape = input_shape
        self.flip_prob = flip_prob
        self.include_com = include_com
        self.include_poi_list = include_poi_list
        self.include_leg_list = include_leg_list
        self.poi_file_ending = poi_file_ending
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_config = transform_config
        self.surface_erosion_iterations = surface_erosion_iterations
        self.save_hyperparameters()

    """
    def prepare_data(
        self,
        bids_surgery_info: BIDS_Global_info,
        save_path: str,
        get_files: callable,
        rescale_zoom: tuple | None = None,
    ):
        master = []
        partial_process_container = partial(
            process_container,
            save_path=save_path,
            rescale_zoom=rescale_zoom,
            get_files=get_files,
        )
        master = pqdm(
            bids_surgery_info.enumerate_subjects(),
            partial_process_container,
            n_jobs=8,
            argument_type="args",
            exception_behaviour="immediate",
        )
        master = [item for sublist in master for item in sublist]
        master_df = pd.DataFrame(master)
        master_df.to_csv(os.path.join(save_path, "master_df.csv"), index=False)
    """

    def setup(self, stage=None):
        self.master_df = pd.read_csv(self.master_df_path)

        # Only keep rows where leg is in "include_leg_list"
        if self.include_leg_list is not None:
            self.master_df = self.master_df[
                self.master_df["leg"].isin(self.include_leg_list)
            ]

        self.train_df = self.master_df[
            self.master_df["subject"].isin(self.train_subjects)
        ]
        self.val_df = self.master_df[self.master_df["subject"].isin(self.val_subjects)]
        self.test_df = self.master_df[
            self.master_df["subject"].isin(self.test_subjects)
        ]

        if self.transform_config is not None:
            transform = [create_transform(self.transform_config)]

        if self.dataset == "Femur":
            self.train_dataset = FemurDataset(
                self.train_df,
                input_shape=self.input_shape,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_leg_list=self.include_leg_list,
                transforms=transform,
                flip_prob=self.flip_prob,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
            )
            self.val_dataset = FemurDataset(
                self.val_df,
                input_shape=self.input_shape,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_leg_list=self.include_leg_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
            )
            self.test_dataset = FemurDataset(
                self.test_df,
                input_shape=self.input_shape,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_leg_list=self.include_leg_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
            )
        elif self.dataset == "Patella":
            self.train_dataset = PatellaDataset(
                self.train_df,
                input_shape=self.input_shape,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_leg_list=self.include_leg_list,
                transforms=transform,
                flip_prob=self.flip_prob,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
            )
            self.val_dataset = PatellaDataset(
                self.val_df,
                input_shape=self.input_shape,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_leg_list=self.include_leg_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
            )
            self.test_dataset = PatellaDataset(
                self.test_df,
                input_shape=self.input_shape,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_leg_list=self.include_leg_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
            )
        elif self.dataset == "LowerLeg":
            self.train_dataset = LowerLegDataset(
                self.train_df,
                input_shape=self.input_shape,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_leg_list=self.include_leg_list,
                transforms=transform,
                flip_prob=self.flip_prob,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
            )
            self.val_dataset = LowerLegDataset(
                self.val_df,
                input_shape=self.input_shape,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_leg_list=self.include_leg_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
            )
            self.test_dataset = LowerLegDataset(
                self.test_df,
                input_shape=self.input_shape,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_leg_list=self.include_leg_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
            )


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class FemurDataModule(POIDataModule):
    def __init__(
        self,
        master_df: PathLike,
        train_subjects: list,
        val_subjects: list,
        test_subjects: list,
        input_shape: tuple = (170, 145, 625),#(190, 215, 1165),
        flip_prob: float = 0.5,
        transform_config: dict | None = None,
        include_com: bool = False,
        include_poi_list=None,
        include_leg_list=None,
        batch_size: int = 1,
        num_workers: int = 0,
        poi_file_ending: str = "poi.json",
        surface_erosion_iterations: int = 1,
    ):
        super().__init__(
            dataset="Femur",
            master_df=master_df,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            input_shape=input_shape,
            flip_prob=flip_prob,
            transform_config=transform_config,
            include_com=include_com,
            include_poi_list=include_poi_list,
            include_leg_list=include_leg_list,
            batch_size=batch_size,
            num_workers=num_workers,
            poi_file_ending=poi_file_ending,
            surface_erosion_iterations=surface_erosion_iterations,
        )

class PatellaDataModule(POIDataModule):
    def __init__(
        self,
        master_df: PathLike,
        train_subjects: list,
        val_subjects: list,
        test_subjects: list,
        input_shape: tuple = (70, 45, 70), #(190, 215, 1165),#(128, 128, 96),
        flip_prob: float = 0.5,
        transform_config: dict | None = None,
        include_com: bool = False,
        include_poi_list=None,
        include_leg_list=None,
        batch_size: int = 1,
        num_workers: int = 0,
        poi_file_ending: str = "poi.json",
        surface_erosion_iterations: int = 1,
    ):
        super().__init__(
            dataset="Patella",
            master_df=master_df,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            input_shape=input_shape,
            flip_prob=flip_prob,
            transform_config=transform_config,
            include_com=include_com,
            include_poi_list=include_poi_list,
            include_leg_list=include_leg_list,
            batch_size=batch_size,
            num_workers=num_workers,
            poi_file_ending=poi_file_ending,
            surface_erosion_iterations=surface_erosion_iterations,
        )

class LowerLegDataModule(POIDataModule):
    def __init__(
        self,
        master_df: PathLike,
        train_subjects: list,
        val_subjects: list,
        test_subjects: list,
        input_shape: tuple = (150, 165, 550),#(190, 215, 1165),#(128, 128, 96),
        flip_prob: float = 0.5,
        transform_config: dict | None = None,
        include_com: bool = False,
        include_poi_list=None,
        include_leg_list=None,
        batch_size: int = 1,
        num_workers: int = 0,
        poi_file_ending: str = "poi.json",
        surface_erosion_iterations: int = 1,
    ):
        super().__init__(
            dataset="LowerLeg",
            master_df=master_df,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            input_shape=input_shape,
            flip_prob=flip_prob,
            transform_config=transform_config,
            include_com=include_com,
            include_poi_list=include_poi_list,
            include_leg_list=include_leg_list,
            batch_size=batch_size,
            num_workers=num_workers,
            poi_file_ending=poi_file_ending,
            surface_erosion_iterations=surface_erosion_iterations,
        )
    """
    def prepare_data(
        self,
        bids_surgery_info: BIDS_Global_info,
        save_path: str,
        rescale_zoom: tuple | None = None,
    ):
        gruber_get_files = partial(
            get_files,
            get_poi=get_gruber_poi,
            get_ct=get_ct,
            get_subreg=get_subreg,
            get_vertseg=get_vertseg,
        )
        super().prepare_data(
            bids_surgery_info, save_path, gruber_get_files, rescale_zoom
        )
    """

def create_data_module(config):
    module_type = config["type"]
    module_params = config["params"]
    if module_type == "FemurDataModule":
        return FemurDataModule(**module_params)
    elif module_type == "PatellaDataModule":
        return PatellaDataModule(**module_params)
    elif module_type == "LowerLegDataModule":
        return LowerLegDataModule(**module_params)
    else:
        raise ValueError(f"Data module type {module_type} not recognized")

    

if __name__ == "__main__":
    bids_surgery_poi = BIDS_Global_info(
        ["/home/daniel/Data/Implants/dataset-implants"], additional_key=["seq"]
    )
    save_path = "/home/daniel/Data/Implants/cutouts_scale-1-1-1"

    data_module = FemurDataModule()#ImplantsDataModule()
    data_module.prepare_data(bids_surgery_poi, save_path, rescale_zoom=(1, 1, 1))

    bids_surgery_poi = BIDS_Global_info(
        ["/home/daniel/Data/Gruber/dataset-poi-gruber"],
        parents=["rawdata", "derivatives_seg_new"],
        additional_key=["seq", "snapshot", "ref"],
    )
    save_path = "/home/daniel/Data/Gruber/cutouts_scale-1-1-1"

    data_module = PatellaDataModule() #GruberDataModule()
    data_module.prepare_data(bids_surgery_poi, save_path, rescale_zoom=(1, 1, 1))
