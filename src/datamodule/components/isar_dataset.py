import glob
import io
import os
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torchvision.transforms as T
from pandas import DataFrame
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor


class ISARDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: Optional[DataFrame] = None,
        build_from_raw_data: bool = False,
        transformation: Optional[Callable] = None,
        csv_path: Optional[str] = None,
        root_dir: Optional[str] = None,
    ):
        super().__init__()
        if csv_path and not dataframe:
            self.dataframe = pd.read_csv(csv_path)
        else:
            self.dataframe = (
                self.build_from_raw_data(root_dir=root_dir) if build_from_raw_data else dataframe
            )
        self.transformation = transformation

    def update_transformation(self, transformation):
        self.transformation = transformation

    def __len__(self):
        return len(self.dataframe)

    def remove_all_item(self):
        self.dataframe = None

    def add_item(self, image_path, label_index):
        self.dataframe = self.dataframe.append(
            {"image_path": image_path, "label_index": label_index}, ignore_index=True
        )

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row["image_path"]
        label = row["label"]
        image = Image.open(image_path).convert("RGB")

        if self.transformation:
            image = self.transformation(image)

        return image, label

    def __plotsample__(self, idx):
        image, label = self.__getitem__(idx)
        if isinstance(image, torch.Tensor):
            image = ToPILImage()(image)  # Remove batch dimension for display
        image = px.imshow(image)
        return image

    def __getmodel__(self, idx, model):
        sample_tensor, sample_gt = self.__getitem__(idx)
        sample_id = self.dataframe.iloc[idx]["id"]

        # Ensure the tensor has the correct dimensions
        sample_tensor = sample_tensor.unsqueeze(1)  # Add batch dimension, shape [1, 1, 256, 512]
        pred_model = model(sample_tensor)
        pred_model = pred_model.mean(dim=0)  # reashpe the tensor from Shape [3,2] to Shape [1, 2]
        pred_model = pred_model.softmax(dim=-1)
        pred_df = pd.DataFrame(pred_model.detach().numpy()).T
        pred_df.columns = self.dataframe["label"].unique()
        pred_fig = px.bar(
            pred_df,
            x=pred_df.columns,
            title=f"{sample_id} Prediction results, The real label: {sample_gt}",
        )
        return pred_fig

    def build_from_raw_data(self, root_dir):
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
            # df["image_path"] = df["id"].apply(lambda id: glob.glob(f"{dir}/*{id}*.png"))
            df["label"] = label
            all_data.append(df)
        combined_df = pd.concat(all_data, ignore_index=True)
        label_to_index = {label: idx for idx, label in enumerate(combined_df["label"].unique())}
        combined_df["label_index"] = combined_df["label"].apply(lambda x: label_to_index[x])
        return combined_df


def load_dataset(self, path="data/ISAR/isar_dataset.csv", transformation=None):
    """Load a dataset from a CSV file.

    Args:
        filename (str): The filename of the CSV file to read.
        transform: Any transformations to apply to the images.
    Returns:
        ISARDataset: The loaded dataset object.
    """
    dataframe = pd.read_csv(path)
    return ISARDataset(dataframe, transformation=transformation)


def save_dataset(
    self,
    dataset,
    filename="isar_dataset.csv",
    save_path="data/ISAR",
):
    """Save the dataset's DataFrame to a CSV file.

    Args:
        dataset (ISARDataset): The dataset object containing the data.
        filename (str): The filename of the CSV file to write.
        save_path (str): The path to save the file in. Defaults to the current directory.
    """
    dataframe = dataset.dataframe
    save_file = os.path.join(save_path, filename)
    dataframe.to_csv(save_file, index=False)
    print(f"Dataset saved to {save_file}")
