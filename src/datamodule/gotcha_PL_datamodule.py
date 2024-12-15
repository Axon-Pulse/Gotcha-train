from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split

from src.datamodule.components.transformation import TransformsWrapper


class GotchaDataModule(LightningDataModule):
    """`LightningDataModule` for the Generic dataset.

    a database of simulated ISAR images

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

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = False,
        datasets: DictConfig = None,
        transforms: DictConfig = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.cfg_datasets: DictConfig = datasets
        self.transforms: DictConfig = transforms

        self.stage: Optional[str] = "Train"
        self.main_dataset: Optional[Dataset] = None

        self.train_set: Optional[Dataset] = None
        self.valid_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None
        self.predict_set: Dict[str, Dataset] = OrderedDict()

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes for this dataset.
        """
        return self.hparams.model.num_classes

    def _get_dataset_(self, split_name: str, dataset_name: Optional[str] = None) -> Dataset:
        transforms = TransformsWrapper(
            self.transforms.get(split_name)
            if self.transforms.get(split_name)
            else self.transforms.get("train")
        )
        cfg = self.cfg_datasets.get(split_name)
        if dataset_name:
            cfg = cfg.get(dataset_name)
        dataset: Dataset = hydra.utils.instantiate(cfg, transformation=transforms)
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
        if self.cfg_datasets.get("train_val_test_split"):
            data_list_names = list(self.cfg_datasets.keys())
            if len(data_list_names) > 2 and not self.cfg_datasets.get("predict"):
                # TODO check if data_list_names have  a predict key
                raise RuntimeError(
                    "applying a train_val_test_split requires a single dataset parameters pass in the <self.cfg_datasets> config, i.e."
                    "datasets: {train_val_test_split: [float, float, float] , some_data_set: [*args] }"
                    "instead cfg_datasets.keys() are:"
                    + str(data_list_names)
                    + "where, cfg_datasets is: "
                    + str(self.cfg_datasets)
                )
            self.main_dataset = self._get_dataset_(
                data_list_names[1]
            )  # TODO: indexing is not recommended data_list_names ,replace with other pulling method
            self.train_set, self.valid_set, self.test_set = random_split(
                dataset=self.main_dataset,
                lengths=self.cfg_datasets.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

        if not self.train_set and not self.valid_set and not self.test_set:
            self.train_set = self._get_dataset_("train")
            self.valid_set = self._get_dataset_("valid")
            self.test_set = self._get_dataset_("test")

        # load predict datasets only if it exists in config
        if (stage == "predict") and self.cfg_datasets.get("predict"):
            for dataset_name in self.cfg_datasets.get("predict").keys():
                self.predict_set[dataset_name] = self._get_dataset_(
                    "predict", dataset_name=dataset_name
                )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.valid_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        NOTE: this dataloder must support batch_size=1, and add_item(ANY) method.

        :return: The predict dataloader.
        """
        loaders = []
        for _, dataset in self.predict_set.items():
            loaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                )
            )
        return loaders


if __name__ == "__main__":
    _ = GotchaDataModule()
