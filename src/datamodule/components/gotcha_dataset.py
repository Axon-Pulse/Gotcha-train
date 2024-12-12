import glob
import os
import pickle
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.io as sio
import torch
from pandas import DataFrame
from torchvision.transforms import ToPILImage

from src.datamodule.components.transformation import TransformsWrapper


class GotchaDataset(torch.utils.data.Dataset):
    """Generic dataset structure."""

    def __init__(
        self,
        dataframe: Optional[DataFrame] = None,
        build_from_raw_data: bool = False,
        transformation: Optional[Callable] = None,
        csv_path: Optional[str] = None,
        root_dir: Optional[str] = None,
        **kargs,
    ):
        """Generic dataset structure.

        Args:
            dataframe (Optional[DataFrame], optional): the dataframe \
                represented in by the dataset. Defaults to None.
            build_from_raw_data (bool, optional): . Defaults to False.
            transformation (Optional[Callable], optional): \
                transformation to be performed on single data sample. Defaults to None.
            csv_path (Optional[str], optional): path to the dataset csv. Defaults to None.
            root_dir (Optional[str], optional): path to the root data directory. Defaults to None.
        """
        super().__init__()
        self.__dict__.update(**kargs)

        if csv_path and not dataframe:
            self.dataframe = pd.read_csv(csv_path)
            with open(csv_path[:-4] + ".pkl", "rb") as fp:  # Unpickling
                self.labels = pickle.load(fp)
        else:
            self.dataframe = (
                self.build_from_raw_data(root_dir=root_dir) if build_from_raw_data else dataframe
            )
        self.transformation = transformation

    def update_transformation(
        self, transformation: Union[torch.nn.Module, TransformsWrapper, Callable]
    ):
        """
        This method updates the list of transforms performed on the data sample
        Args:
            transformation (Union[torch.nn.Module, TransformsWrapper,Callable]): \
                the additional transformation to be preformed
        """

        self.transformation = transformation

    def __len__(self):
        """
        gets the length of the dataset
        Returns:
            int: length of the dataset
        """
        return len(self.dataframe)

    def remove_all_item(self):
        """Function to remove all data in the dataset."""
        self.dataframe = None

    def add_item(self, image_path: Union[os.PathLike], label_index: Any):
        """Adds single sample to the dataset dataframe.

        Args:
            image_path (os.PathLike): path to the additional datasample
            label_index (Any): _description_
        """
        self.dataframe = self.dataframe.append(
            {"image_path": image_path, "label_index": label_index}, ignore_index=True
        )

    def __getitem__(self, idx: int):
        """The main task of the dataset class.

        Args:
            idx (int): int ~ U[0,len(self.dataset)-1]
        """
        row = self.dataframe.iloc[idx]

        # pull and process single sample and label
        raw_sample_path = row[self.dataset_keys.df_raw_data_sample_key]

        raw_sample = self.get_sample_from_path(raw_sample_path)
        raw_sample = torch.as_tensor(raw_sample)
        # Move detections to batch dimension
        # raw_sample = np.expand_dims(raw_sample, axis=1)

        if self.transformation:
            raw_sample = self.transformation(raw_sample)

        label_dict = {"label": np.array(self.labels[idx]) * 1.0}

        return raw_sample, label_dict

    def __plotsample__(self, idx):
        image, label = self.__getitem__(idx)
        if isinstance(image, torch.Tensor):
            image = ToPILImage()(image)
        # image = ToPILImage()(image)
        print(type(image))
        image = px.imshow(image)
        return image

    def get_sample_from_path(
        self, sample_path: os.PathLike, file_extention: str = "npy"
    ):  # TODO:move to utils
        if file_extention == "npy":
            try:
                raw_sample = np.load(sample_path).astype(np.float32)
            except Exception as ex:
                print(f"couldnt load npy file in {sample_path}")
        elif file_extention == "mat":
            try:
                raw_sample = sio.loadmat(sample_path)
            except Exception as ex:
                print(f"couldnt load mat file in {sample_path}")
        else:
            print(f"unsupported file format for {sample_path}")
        return raw_sample

    def build_from_raw_data(self, root_dir: os.PathLike):
        all_data = []
        label_dirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]

        for label_dir in label_dirs:
            label = os.path.basename(label_dir)
            excel_file = glob.glob(os.path.join(label_dir, "*.xlsx"))[
                0
            ]  # Assumes one Excel file per label directory
            df = pd.read_excel(excel_file)
            df["image_path"] = df["id"].apply(
                lambda id: glob.glob(os.path.join(label_dir, f"*{id}*.png"))[0]
            )
            df["label"] = label
            all_data.append(df)
        combined_df = pd.concat(all_data, ignore_index=True)
        label_to_index = {label: idx for idx, label in enumerate(combined_df["label"].unique())}
        combined_df["label_index"] = combined_df["label"].apply(lambda x: label_to_index[x])
        return combined_df


# def load_dataset(self, path="data/ISAR/isar_dataset.csv", transformation=None):
#     """Load a dataset from a CSV file.

#     Args:
#         filename (str): The filename of the CSV file to read.
#         transform: Any transformations to apply to the images.
#     Returns:
#         ISARDataset: The loaded dataset object.
#     """
#     dataframe = pd.read_csv(path)
#     return ISARDataset(dataframe, transformation=transformation)


# def save_dataset(
#     self,
#     dataset,
#     filename="isar_dataset.csv",
#     save_path="data/ISAR",
# ):
#     """Save the dataset's DataFrame to a CSV file.

#     Args:
#         dataset (ISARDataset): The dataset object containing the data.
#         filename (str): The filename of the CSV file to write.
#         save_path (str): The path to save the file in. Defaults to the current directory.
#     """
#     dataframe = dataset.dataframe
#     save_file = os.path.join(save_path, filename)
#     dataframe.to_csv(save_file, index=False)
#     print(f"Dataset saved to {save_file}")


if __name__ == "__main__":
    _ = GotchaDataset()
    pass
