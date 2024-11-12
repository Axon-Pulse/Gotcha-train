# The provided code defines a neural network model called IsarClassifier for image classification
# tasks using a CNN encoder and a classification head with training and validation steps implemented
# for PyTorch Lightning.
import sys
from typing import Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.simple_autocoder import ClassificationHead, Encoder


class IsarClassifier(LightningModule):
    def __init__(
        self,
        input_shape: Tuple,  # (1,256,512),
        optimizer: torch.optim.Optimizer,
        number_of_classes: int,  # 2,
        net: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.trainer_data_module = None

        self.n_classes = number_of_classes
        self.input_shape = tuple(input_shape)
        self.encoder = Encoder(input_channels=input_shape[0])
        cls_head_input_size = self.encoder(torch.rand((1,) + tuple(input_shape))).numel()
        self.net = ClassificationHead(
            input_dim=cls_head_input_size, n_classes=self.n_classes, p_dropout=0.3
        )
        self.loss_func = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=self.n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.n_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    @property
    def label_to_class_name(self) -> dict:
        """Get the number of classes.

        :return: a mapping from the class labels to the class names
        """
        return {0: "A69", 1: "Brunswhik"}

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def forward(self, x):
        # print("ClassificationHead Input shape:", x.shape)
        x = self.encoder(x)
        # print("ClassificationHead After encoder:", x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        # print("ClassificationHead After flattening:", x.shape)
        return self.net(x)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of datamodule.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)
        self.train_acc.update(preds, targets)
        self.train_loss.update(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """

        Step function called during ~lightning.pytorch.trainer.trainer.Trainer.predict.
        By default, it calls ~lightning.pytorch.core.LightningModule.forward.
        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
            dataloader_idx (int, optional): _description_. Defaults to 0.

        Returns:
            a tensor of prediction
        """
        x_single, _ = batch
        return self(x_single)

    def embeddings(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # x.flatten(start_dim=1)


if __name__ == "__main__":
    _ = IsarClassifier(input_shape=(1, 256, 512), number_of_classes=2)
    # x = torch.rand((4,1, 256, 512))
    # logits = model(x)
