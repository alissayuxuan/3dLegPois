import os
from functools import partial
from os import PathLike
from typing import TypeVar

import pandas as pd
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

from dataset.dataset import FemurDataset, PatellaDataset, LowerLegDataset, PoiDataset
from transforms.transforms import create_transform

PoiType = TypeVar("PoiType", bound=PoiDataset)


class POIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        master_df: PathLike,
        train_subjects: list,
        val_subjects: list,
        test_subjects: list,
        input_shape: tuple = (190, 213, 280),
        zoom: tuple = (0.8, 0.8, 0.8),
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
        self.zoom = zoom
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
                zoom=self.zoom,
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
                zoom=self.zoom,
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
                zoom=self.zoom,
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
                zoom=self.zoom,
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
                zoom=self.zoom,
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
                zoom=self.zoom,
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
                zoom=self.zoom,
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
                zoom=self.zoom,
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
                zoom=self.zoom,
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
        input_shape: tuple = (150, 215, 115),
        zoom: tuple = (0.8, 0.8, 0.8),
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
            zoom=zoom,
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
        input_shape: tuple = (150, 215, 185),
        zoom: tuple = (0.8, 0.8, 0.8),
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
            zoom=zoom,
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
        input_shape: tuple = (150, 215, 95),
        zoom: tuple = (0.8, 0.8, 0.8),
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
            zoom=zoom,
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
