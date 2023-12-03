import glob
import os
import re
from typing import List, Dict, Tuple, Union

from flax.training import checkpoints
from omegaconf import OmegaConf

from pli.inference.abc.types import ABCTrainState
from pli.inference.pseudo_likelihood_inference.types import PLITrainState
from pli.inference.snpe.types import SNPETrainState
from pli.models.particle_samplers.smc_sampler import build_smc_sampler
from pli.models.utils import (
    build_density_estimator,
    build_conditioned_density_estimator,
)


def get_checkpoint_names(
    log_dir, ckpt_name="checkpoint", sort_by="step"
) -> Tuple[List[str], Dict]:
    files = glob.glob(
        os.path.join(
            log_dir,
            ckpt_name + "*",
        )
    )
    ckpt_step = [int(re.search("checkpoint_(\d+)", file).group(1)) for file in files]
    if sort_by == "step":
        ckpt_step, files = (list(t) for t in zip(*sorted(zip(ckpt_step, files))))
    info = dict(steps=ckpt_step)
    return files, info


def load_ckpt(
    cfg, step: int = None
) -> Union[ABCTrainState, PLITrainState, SNPETrainState]:
    if cfg.method.name == "ABC":
        state = ABCTrainState()
    elif cfg.method.name == "PLI":
        state = PLITrainState(None)
    elif cfg.method.name == "SNPE":
        state = SNPETrainState(None)
    else:
        raise ValueError(f"Unknown experiment name: {cfg.experiment.name}")

    if step is None:
        state = checkpoints.restore_checkpoint(cfg.logs_dir, state)
    else:
        state = checkpoints.restore_checkpoint(cfg.logs_dir, state, step=step)
    return state


def load_model_from_ckpt(
    cfg, param_support, param_sample, observation, step: int = None
):
    model_cfg = OmegaConf.to_container(cfg.method.model).copy()
    model_name = model_cfg.pop("name")
    if cfg.method.name == "ABC":
        model = build_smc_sampler(param_support, **cfg.method.model)
    elif cfg.method.name == "PLI":
        model = build_density_estimator(
            model_name, param_support, param_sample, **model_cfg
        )
    elif cfg.method.name == "SNPE":
        model = build_conditioned_density_estimator(
            model_name, param_support, param_sample, observation, **model_cfg
        )
    else:
        raise ValueError(f"Unknown experiment name: {cfg.experiment.name}")
    state = load_ckpt(cfg, step)
    return state, model
