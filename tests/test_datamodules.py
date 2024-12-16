import os
import sys
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from lightning import LightningDataModule
from omegaconf import DictConfig, open_dict

from src.datamodule.gotcha_PL_datamodule_v2 import GotchaDataModule
from src.datamodule.mnist_datamodule import MNISTDataModule


def test_gotcha_datamodule() -> None:
    """Tests `GotchaDataModule` to verify that it can be downloaded correctly,
    that the necessary attributes were created (e.g., the dataloader objects),
    and that dtypes and batch sizes correctly match.
    """
    # Initialize Hydra and compose the configuration directly inside the test
    with initialize(version_base="1.3", config_path="../configs/datamodule"):
        cfg = compose(
            config_name="gotcha_v2.yaml",
            return_hydra_config=True,
            overrides=[
                "datasets.main.csv_path=tests/data/gotcha/gt_with_recs_stats_df_subset_for_test.csv",
            ],
        )
    # Now use the configuration to test the data module
    dm = GotchaDataModule(
        batch_size=2,
        num_workers=4,
        pin_memory=False,
        datasets=cfg.datasets,
        train_val_split=[0.8, 0.2],
    )

    # Perform assertions or checks on the data module here
    # Example: checking if the dataloaders were correctly created
    dm.setup()
    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "datamodule/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
