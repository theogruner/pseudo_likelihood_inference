"""Run an SBI experiment."""
import os
import time
import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
from matplotlib import pyplot as plt
import numpy as np
import wandb


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.gpu:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

    import jax
    import jax.numpy as jnp
    from flax.training import checkpoints

    import pli.tasks
    from pli.utils.casting import flatten_dict
    from pli.utils.dataloaders import build_dataloader, Dataset
    from pli.inference import build_pli_experiment, build_abc_experiment, build_snpe_experiment
    from pli.tasks import Task

    # Set-up task
    cfg.task.n_eval_data = cfg.task.n_train_data
    cfg.method.n_samples_per_param = cfg.task.n_train_data

    task_ = getattr(pli.tasks, cfg.task.name)
    task: Task = task_(**cfg.task)

    # Set data directory
    if cfg.data_dir is None:
        cfg.data_dir = os.path.join(pli.DATA_DIR, task.data_dir, f"seed_{cfg.method.seed}")

    (
        _,
        prior,
        simulator,
        posterior_samples,
        param_support,
        train_data,
        eval_data,
    ) = task.initialize_task(
        cfg.data_dir, seed=cfg.method.seed
    )

    train_dataset = Dataset.create(train_data)
    eval_dataset = Dataset.create(eval_data)

    # Overkill as we currently only assume data of size 1.
    # When extending to amortized PLI, we need to change this.
    train_dataloader = build_dataloader(train_dataset, batch_size=train_dataset.size)
    eval_dataloader = build_dataloader(eval_dataset, batch_size=eval_data.size)

    rng_key = jax.random.PRNGKey(cfg.method.seed)
    rng_key, next_rng_key = jax.random.split(rng_key)
    sample_data = train_dataloader(rng_key)

    model_cfg = OmegaConf.to_container(cfg.method.model).copy()
    model_cfg.pop("name")

    # Initialize experiment
    if cfg.method.name == "PLI":
        build_sbi_method = build_pli_experiment
        pseudo_likelihood_cfg = OmegaConf.to_container(cfg.method.pseudo_likelihood).copy()
        pseudo_likelihood_cfg.pop("name")
        method_specific_dict = dict(
            epsilon=cfg.method.epsilon,
            nu=cfg.method.nu,
            pseudo_likelihood_name=cfg.method.pseudo_likelihood.name,
            pseudo_likelihood_cfg=pseudo_likelihood_cfg,
            statistical_distance=cfg.method.statistical_distance.name
        )
    elif cfg.method.name == "ABC":
        build_sbi_method = build_abc_experiment
        pseudo_likelihood_cfg = OmegaConf.to_container(cfg.method.pseudo_likelihood).copy()
        pseudo_likelihood_cfg.pop("name")
        method_specific_dict = dict(
            pseudo_likelihood_name=cfg.method.pseudo_likelihood.name,
            pseudo_likelihood_cfg=pseudo_likelihood_cfg,
            statistical_distance=cfg.method.statistical_distance.name
        )
    elif cfg.method.name == "SNPE":
        build_sbi_method = build_snpe_experiment
        method_specific_dict = {}
    else:
        raise ValueError(f"Experiment {cfg.method.name} not supported.")

    experiment = build_sbi_method(
        rng_key=next_rng_key,
        model_name=cfg.method.model.name,
        param_support=param_support,
        prior=prior,
        simulator=simulator,
        sample_data=sample_data,
        n_param_samples=cfg.method.n_param_samples,
        n_samples_per_param=cfg.method.n_samples_per_param,
        model_cfg=model_cfg,
        max_resampling_tries=cfg.method.max_resampling_tries,
        sampling_batch_size=cfg.method.sampling_batch_size,
        n_parallel_operations=cfg.method.n_parallel_operations,
        **method_specific_dict,
    )

    # Set-up logging
    tags = [
        cfg.task.name,
        cfg.method.model.name,
        f"n_param_samples_{cfg.method.n_param_samples}",
        f"n_samples_per_param_{cfg.method.n_samples_per_param}",
    ] + [
            f"{key}_{val}"
            for key, val in cfg.method.model.items()
            if key != "name"
    ]
    if cfg.method.name in ["PLI", "ABC"]:
        tags += (
            [
                cfg.method.pseudo_likelihood.name,
                cfg.method.statistical_distance.name,
            ]
            + [
                f"{key}_{val}"
                for key, val in cfg.method.pseudo_likelihood.items()
                if key != "name"
            ]
        )
    logger = None
    if cfg.saving:
        # Set saving directory
        logging_name = None
        if cfg.logs_dir is None:
            time_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            log_dir = cfg.group
            if cfg.method.seed is not None:
                log_dir = os.path.join(log_dir, f"seed_{cfg.method.seed}")
                logging_name = (
                    f"seed-{cfg.method.seed}-{cfg.method.model.update_method}"
                    f"-n_param_samples_{cfg.method.n_param_samples}"
                    f"-n_samples_per_param_{cfg.method.n_samples_per_param}"
                    f"-{time_str}"
                )
            cfg.logs_dir = os.path.join(pli.LOGS_DIR, log_dir, logging_name)

        flattened_cfg = flatten_dict(cfg)
        os.makedirs(cfg.logs_dir, exist_ok=True)
        logger = wandb.init(
            project=cfg.project,
            group=cfg.group,
            name=logging_name,
            config=flattened_cfg,
            dir=cfg.logs_dir,
            tags=tags,
            settings=wandb.Settings(start_method="fork"),
        )
        with open(os.path.join(cfg.logs_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)
    figures = task.task_specific_plots()

    # Initialize experiment
    rng_key, next_rng_key = jax.random.split(rng_key)
    train_state, _ = experiment.init(rng_key)

    # run experiment
    print("Starting experiment:")
    print(OmegaConf.to_yaml(cfg))
    acc_train_walltime = 0.
    acc_eval_walltime = 0.
    for episode in range(1, cfg.method.episodes + 1):
        rng_key, next_rng_key = jax.random.split(train_state.rng_key)
        train_state = train_state.replace(rng_key=next_rng_key)
        batched_train_data = train_dataloader(rng_key)
        t = time.time()
        if cfg.debug:
            train_log_list = []
            for data in batched_train_data:
                train_state, tl = experiment.step(train_state, data)
                train_log_list.append(tl)
            train_log = {}
            for key in train_log_list[0].keys():
                train_log[key] = jnp.stack([tl[key] for tl in train_log_list])
        else:
            train_state, train_log = jax.lax.scan(
                experiment.step, train_state, batched_train_data
            )
        if train_log:
            train_log = jax.tree_util.tree_map(lambda x: np.reshape(x, -1), train_log)
        train_walltime = time.time() - t
        acc_train_walltime += train_walltime

        # Evaluate
        if episode % cfg.method.eval_interval == 0:
            rng_key, next_rng_key = jax.random.split(train_state.rng_key)
            train_state = train_state.replace(rng_key=next_rng_key)
            batched_eval_data = eval_dataloader(rng_key)

            t = time.time()
            eval_log_list, eval_param_samples, eval_simulations = [], [], []
            for data in batched_eval_data:
                rng_key, next_rng_key = jax.random.split(train_state.rng_key)
                train_state = train_state.replace(rng_key=next_rng_key)
                p_samples, sims, tl = experiment.evaluate(
                    rng_key,
                    train_state,
                    data,
                    cfg.method.n_posterior_samples,
                    param_support,
                    n_simulation_samples=1,
                    ground_truth_params=task.ground_truth_parameters(),
                    posterior_samples=posterior_samples
                )
                eval_log_list.append(tl)
                eval_param_samples.append(p_samples)
                eval_simulations.append(sims)
            eval_logs = {}
            for key in eval_log_list[0].keys():
                eval_logs[key] = jnp.stack([tl[key] for tl in eval_log_list])
            eval_logs = jax.tree_util.tree_map(
                lambda x: np.asarray(jnp.mean(x)), eval_logs
            )
            eval_walltime = time.time() - t
            acc_eval_walltime += eval_walltime
            eval_logs["epoch"] = episode
            eval_logs["simulations"] = int(train_state.n_simulations)
            eval_logs["acc train walltime"] = acc_train_walltime
            eval_logs["train walltime"] = train_walltime
            eval_logs["acc eval walltime"] = acc_eval_walltime
            eval_logs["eval walltime"] = eval_walltime
            eval_logs["total runtime"] = acc_train_walltime + acc_eval_walltime

            print(f"-------------------Epoch-{episode}-------------------")
            for key, val in eval_logs.items():
                if isinstance(val, int):
                    print(f"{key:<25} {val}")
                else:
                    print(f"{key:<25} {val:.3f}")

            # Plot relevant plots in the final episode
            if cfg.plotting:
                plot_data = dict(
                    params=np.asarray(eval_param_samples[0]),
                    reference_params=task.ground_truth_parameters()[np.newaxis, :],
                    posterior_params=np.asarray(posterior_samples)
                    if posterior_samples is not None else None,
                    observations=np.asarray(eval_simulations[0]),
                    targets=np.asarray(batched_eval_data[0]),
                )
                for fig in figures:
                    fig.update(plot_data)
                if logger:
                    eval_logs.update({fig.name: wandb.Image(fig.fig) for fig in figures})

            if logger:
                logger.log(eval_logs)
                if train_log:
                    n_train_log_steps = len(list(train_log.values())[0])
                    for i in range(n_train_log_steps):
                        logger.log(
                            {key: np.asarray(val[i]) for key, val in train_log.items()}
                        )

        if episode % cfg.method.checkpoint_interval == 0 and cfg.saving:
            checkpoints.save_checkpoint(
                cfg.logs_dir,
                train_state,
                episode,
                keep_every_n_steps=cfg.method.checkpoint_interval,
                overwrite=True,
            )

    if cfg.plotting:
        for fig in figures:
            fig.show()
        plt.show()

    print("Finished experiment!")
    wandb.finish()


if __name__ == "__main__":
    main()
