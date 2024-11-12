import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import torch
from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def get_label_map_from_labels_dict(
    output_shape: Tuple, labels_dict: Dict, h_tolerance: int = 1, w_tolerance: int = 1
) -> Any:
    """
    assembles labels map with 0-1 scores for ground truth depends on rg,dg locations
    assuming shape=[...,doppler_len,range_len]
    Args:
        output_shape (Tuple): the shape of the output
        labels_dict (Dict): dictionary contains 'dg' and 'rg' keys
        h_tolerance (int, optional): determines the h dimension label shape. Defaults to 1.
        w_tolerance (int, optional): determines the w dimension label shape. Defaults to 1.

    Returns:
        Any:  0-1 tensor with the same shape as output shape
    """
    output = torch.zeros(output_shape)
    for _, row in pd.DataFrame([labels_dict]).iterrows():
        output[
            ...,
            row["dg"] - h_tolerance // 2 : row["dg"] + h_tolerance // 2 + 1,
            row["rg"] - w_tolerance // 2 : row["rg"] + w_tolerance // 2 + 1,
        ] = 1
    return output


def get_roc_plot(roc_results) -> go.Figure:
    """
    utility function to plot torchmetrics.BinaryROC results
    Args:
        roc_results (_type_): iterable  [fpr_vector,tpr_vector,ths_vector]

    Returns:
        plotly.graph_objects.Figure: roc curve and thresholds figure
    """
    fpr, tpr, ths = roc_results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=fpr, y=ths, mode="lines", name="Thresholds"))
    fig.update_layout(xaxis_type="log")
    fig.update_xaxes(title="fpr [logscale]")
    fig.update_yaxes(title="tpr")
    return fig
