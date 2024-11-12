from typing import Any, Dict, List, Tuple

import os
import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from dash import Dash, html, dcc, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    BasicGradioApp,
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from visualizer import main_layout,main_callbacks

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def prediction(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing! prediction")
    # _ = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    datamodule.setup(stage="predict")

    metric_dict = trainer.callback_metrics
    # print(f'the dataset is: {datamodule.cfg_datasets.main}')
    visualizer_instance = main_layout.Visualization(dataset=datamodule.predict_dataloader()[0].dataset,
                                                    model=model, 
                                                    transforms=cfg.transformation,
                                                    default_transforms = cfg.datamodule.transforms,
                                                    # predict_transform = cfg.datamodule.transforms.valid_test_predict
                                                    )

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
    app.layout = html.Div(id="default-view", 
            children=[
            html.Img(src='/assets/axon_logo.png', style={'height': '150px', 'width': '150px'}),
            html.H1('Visualizator', style={'textAlign': "center"}),
            html.Hr(),
            visualizer_instance.default_layout(),
    # change the layout of the page based on the selected option:
            html.Div(id='page-content'),
            ])
    app.css.append_css({
        'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
    })

    app.css.append_css({
        'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
    })

    app.css.append_css({
        'external_url': 'https://codepen.io/chriddyp/pen/brPBPO.css'
    })


    main_callbacks.visualizer_callbacks(app, visualizer_instance)
    app.run_server(debug=True)

        # if cfg.lunch_gradio_app:
        #     BasicGradioApp(model, datamodule, trainer, csv_path=cfg.paths.predict_data_csv_path)
    return metric_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="visualizer.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    prediction(cfg)


if __name__ == "__main__":
    main()
