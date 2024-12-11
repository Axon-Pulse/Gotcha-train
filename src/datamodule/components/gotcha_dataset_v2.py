import glob
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.io as sio
import torch
from pandas import DataFrame
import random
from torchvision.transforms import ToPILImage
import pickle
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
            with open(csv_path[:-4] + ".pkl", "rb") as fp:   # Unpickling
                self.raw_labels = pickle.load(fp)
        else:
            self.dataframe = (
                self.build_from_raw_data(root_dir=root_dir) if build_from_raw_data else dataframe
            )
        self.transformation = transformation
        self.data = []
        self.labels = []
        for idx, row in self.dataframe.iterrows():
            raw_sample_path = row[self.dataset_keys.df_raw_data_sample_key]
        
            self.data.extend([{"path": raw_sample_path, "vector_index": i} for i in range(row["numDetections"])])
            self.labels.extend(self.raw_labels[idx].tolist())
        # remove some of the false elements
        #  Get indices of all `False` elements
        boolean_list = np.array(self.labels)
        other_list = np.array(self.data)
        # Get indices of all `False` elements
        false_indices = np.where(boolean_list == False)[0]

        # Randomly select a quarter of `False` indices to remove
        num_to_remove = int(len(false_indices) // 1.0001)
        indices_to_remove = np.random.choice(false_indices, num_to_remove, replace=False)

        # Use boolean masking to filter the lists efficiently
        mask = np.ones(len(boolean_list), dtype=bool)
        mask[indices_to_remove] = False
        
        # # Apply the mask to both lists
        self.labels = boolean_list[mask].tolist()  # Convert back to list
        self.data = other_list[mask].tolist()  # Convert back to list

        true_labels_count = sum(self.labels)
        self.labels_count = np.array([len(self.labels) - true_labels_count, true_labels_count])
        

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
        return len(self.data)

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
        vector_dict = self.data[idx]
       
    
        raw_sample = self.get_sample_from_path(vector_dict["path"])[vector_dict["vector_index"]]
        raw_sample = torch.as_tensor(raw_sample)[6:]
        # Move detections to batch dimension
        # raw_sample = np.expand_dims(raw_sample, axis=1)

        if self.transformation:
            raw_sample = self.transformation(raw_sample)
        
        
        label = np.array(self.labels[idx])*1.0

        return raw_sample, label

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



if __name__ == "__main__":
    _ = GotchaDataset()
    pass
