from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
import numpy as np
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler

from src.datamodule.components.transformation import TransformsWrapper


class GotchaDataModule(LightningDataModule):
    """`LightningDataModule` for the Gotcha dataset.

    a database of gotcha images with the following structure:

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_setloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.


    """

    def __init__(
        self,
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = False,
        datasets: DictConfig = None,
        transforms: DictConfig = None,
        train_val_split: list = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.cfg_datasets: DictConfig = datasets
        self.transforms: DictConfig = transforms
        self.train_val_split = train_val_split
        self.stage: Optional[str] = "Train"
        self.main_dataset: Optional[Dataset] = None

        self.train_set: Optional[Dataset] = None
        self.valid_set: Optional[Dataset] = None
        self.predict_set: Dict[str, Dataset] = OrderedDict()


    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes for this dataset.
        """
        return self.hparams.model.num_classes

    def _get_dataset_(self, split_name: str, dataset_name: Optional[str] = None) -> Dataset:


        dataset: Dataset = hydra.utils.instantiate(self.cfg_datasets[dataset_name])
        return dataset

    def prepare_data(self):
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.train_set`, `self.valid_set`, `self.test_set`,
        `self.predict_set`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # loading an already process dataset


        self.main_dataset: Dataset = hydra.utils.instantiate(self.cfg_datasets['main'])  
        train_size = int(self.train_val_split[0] * len(self.main_dataset))
        val_size = len(self.main_dataset) - train_size
        self.train_subset, self.val_subset = random_split(self.main_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
        self.test_set = None
        self.train_subset.dataset.transformation = TransformsWrapper(self.transforms.get("train"))  # Apply train transforms to the train dataset
        self.val_subset.dataset.transformation  = TransformsWrapper(self.transforms.get("validation") )
 

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.cfg_datasets['main'].get("weighted_sampling"):
            subset_labels = np.array(self.main_dataset.labels)[self.train_subset.indices]
            true_labels_count = sum(subset_labels)
            class_counts = np.array([len(subset_labels) - true_labels_count, true_labels_count])
            
            # Assign weights to each sample
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[int(label)] for label in subset_labels]
            # Create the sampler
            self.sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True
            )
        return DataLoader(
            dataset=self.train_subset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=self.sampler if self.cfg_datasets['main'].get("weighted_sampling") else None,
            shuffle=True if not self.cfg_datasets['main'].get("weighted_sampling") else False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if self.cfg_datasets['main'].get("weighted_sampling"):
            subset_labels = np.array(self.main_dataset.labels)[self.val_subset.indices]
            true_labels_count = sum(subset_labels)
            class_counts = np.array([len(subset_labels) - true_labels_count, true_labels_count])
            
            # Assign weights to each sample
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[int(label)] for label in subset_labels]
            # Create the sampler
            self.sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True,
            )
        return DataLoader(
            dataset=self.val_subset,
            batch_size=self.hparams.batch_size,
            sampler=self.sampler if self.cfg_datasets['main'].get("weighted_sampling") else None,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True if not self.cfg_datasets['main'].get("weighted_sampling") else False,
        )



if __name__ == "__main__":
    _ = GenericDataModule()
