from typing import Dict, Tuple, Union

import plotly.graph_objects as go
import plotly.express as px
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from lightning import LightningModule
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric,Accuracy
from torchmetrics.classification import ConfusionMatrix

class GotchaPLModule(LightningModule):
    """Pytorch Lightning module for general use."""

    def __init__(
        self,
        input_shape: Tuple,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        **kargs
    ) -> None:
        """
        Args:
            input_shape (Tuple): single sample input shape, expected to be batched.
            model (torch.nn.Module): the used model to be used by the module for all steps.
            loss (torch.nn.Module): the loss to be used.
            optimizer (torch.optim.Optimizer): the optimizer instance to be used.
            scheduler (torch.optim.lr_scheduler, optional): scheduler to be used in the p. Defaults to None.
            compile (bool, optional): determines if the model should be compiled. Defaults to False.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.input_shape = tuple(input_shape)
        # set model
        assert model is not None, "model is required, but not supplied"
        self.model = model
        # set loss
        assert loss is not None, "loss is required, but not supplied"
        self.loss_func = loss
        if scheduler is not None:
            self.scheduler = scheduler
        # simply set all kargs as attributes
        self.__dict__.update(kargs)
        # Optimizer
        assert optimizer is not None, "optimizer is required, but not supplied"
        # Metrics
        assert (
            "metric" in kargs.keys()
        ), "metrics (for train, val and test) are required,\
              but not supplied"
        self.__dict__.update(kargs["metric"])


        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_cm = ConfusionMatrix(task="binary")
        self.train_cm = ConfusionMatrix(task="binary")  # Update num_classes as needed



    # ******************************************************************************
    #
    #                       Model
    #
    # ******************************************************************************

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward flow of the network.

        Args:
            x (torch.Tensor): input batch in dim: [N C_in H_in W_in]

        Returns:
            torch.Tensor: output batch in dim: [N C_out H_out W_out]
        """
        return self.model(x)

    def preprocess_batch(
        self, input_batch: Union[torch.Tensor, Dict]
    ) -> Union[torch.Tensor, Dict]:
        """Utility function to preprocess the batch used in order to keep the network flow clean.

        Args:
            input_batch (Union[torch.Tensor,Dict]): the input batch

        Returns:
            Union[torch.Tensor,Dict]: cleaned,processed version of the input batch
        """
        return input_batch

    def postprocess_batch(
        self, output_batch: torch.Tensor, labels: Union[torch.Tensor, Dict] = None
    ) -> Dict:
        """Utility function for postprocessing the batch used in order to keep network flow clean.

        Args:
            output_batch (torch.Tensor): the model's output batch dim: [N ...]
            labels (Union[torch.Tensor,Dict]): the labels used for the model loss\
                  and evaluation dim: [N ...]

        Returns:
            Dict: postprocessed batch dim: [N ...]
        """
        output_batch = torch.squeeze(output_batch)
        return output_batch

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        """Perform a single model step on a batch of datamodule.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of model_output.
            - A tensor of target labels.
        """
        x, y = batch
        x = self.preprocess_batch(x)
        model_output_raw = self.forward(x)
        model_output_processed = self.postprocess_batch(output_batch=model_output_raw, labels=y)
        loss = self.loss_func(model_output_processed, y)
        return loss, model_output_processed, y

    # ******************************************************************************
    #
    #                       Training
    #
    # ******************************************************************************

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

        
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, model_output, targets = self.model_step(batch)
        self.train_loss.update(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_acc(model_output, targets)
        self.train_cm.update(model_output, targets)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ******************************************************************************
    #
    #                       Validation
    #
    # ******************************************************************************

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, model_output, targets = self.model_step(batch)
        # update and log metrics ## daniel
        if self.val_metric:
            # val_metric_results = self.val_metric(model_output.detach().cpu(), targets.detach().cpu().int())
            self.val_metric.update(model_output.detach().cpu(), targets.detach().cpu().int())

            # if self.get_metric_plot_func:
            #     val_metric_plot = self.get_metric_plot_func(val_metric_results)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_cm.update(model_output, targets)
        self.val_acc(model_output, targets)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)


        if(batch_idx % 1000==0):
            # self.log_samples(batch[0][0][:50])
            self.log_samples(batch[0][batch[1]==1].T,'positive')
            self.log_samples(batch[0][batch[1]==0].T,'negative')
            # self.log_samples(batch[0][0][batch[1]['label']==1][:,6:].T,'positive')
            # self.log_samples(batch[0][0][batch[1]['label']==0][:50,6:].T,'negative')

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."


        acc = self.val_acc.compute()  # get current val acc
        
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        val_cm = self.val_cm.compute()
        train_cm = self.train_cm.compute()
        self.plot_cm(val_cm, "val")
        self.plot_cm(train_cm, "train")
        
        self.val_cm.reset()
        self.train_cm.reset()
        val_metric_results = self.val_metric.compute()
        if self.get_metric_plot_func:
                val_metric_plot = self.get_metric_plot_func(val_metric_results)
                self.logger.experiment.log({"Roc": val_metric_plot})
    # ******************************************************************************
    #
    #                       Test
    #
    # ******************************************************************************

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, model_output, targets = self.model_step(batch)

        # update and log metrics
        if self.test_metric:
            test_metric_results = self.test_metric(model_output, targets.int())
            if self.get_metric_plot_func:
                test_metric_plot = self.get_metric_plot_func(test_metric_results)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    # ******************************************************************************
    #
    #                       Predict
    #
    # ******************************************************************************

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

    # ******************************************************************************
    #
    #                       Setup
    #
    # ******************************************************************************

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    # ******************************************************************************
    #
    #                       Configuration
    #
    # ******************************************************************************

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

    def log_samples(self,sample,name='samples'):
        fig = px.imshow(sample.detach().cpu().T)
        self.logger.experiment.log({name: fig})

    def plot_cm(self,cm, name):
        # Convert confusion matrix to a more readable format if needed
        cm = cm.cpu().numpy()
        
        # Log confusion matrix to W&B
        class_names = ["False_t", "True_t"]

        # Prepare the table for logging


        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 50}, xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")

        # Log the confusion matrix image with epoch
        self.logger.experiment.log({f"{name}/confusion_matrix": wandb.Image(plt)})


if __name__ == "__main__":
    # _ = GenericPLModule()
    pass
