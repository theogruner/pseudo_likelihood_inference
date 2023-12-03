# Pseudo-Likelihood Inference (NeurIPS 2023)

---
[Gruner, T., Belousov, B., Muratore, F., Palenicek, D., & Peters, J. (2023). Pseudo-Likelihood Inference. In Thirty-seventh Conference on Neural Information Processing Systems.](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/TheoGruner/pli_preprint)

### Official repository to repeat the experiments of the paper
We provide [JAX](https://jax.readthedocs.io/en/latest/) implementations of three simulation-based inference methods,
including pseudo-likelihood inference.
The following algorithms are provided in this repository:
- Pseudo-Likelihood Inference (PLI, Ours) [[1]](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/TheoGruner/pli_preprint)
- Sequential Monte Carlo Approximate Bayesian Computation (SMC-ABC) [[2]](https://arxiv.org/abs/1111.1308)
- Automatic Posterior Transformation (APT) [[3]](https://arxiv.org/abs/1905.07488)

### Installing
Run `pip install -e .` to install the package. We tested the code with python 3.9.13.
We highly recommend GPU support for JAX by following the [instructions](https://github.com/google/jax#installation).
For example, to install jax with cuda 12 or 11, either run

```
pip install --upgrade "jaxlib[cuda12_pip]"==0.4.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

or

```
pip install --upgrade "jaxlib[cuda11_pip]"==0.4.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

before or after installing pli.

### Run an experiment
To start an experiment, simply run:
```
python scripts/run_sbi_experiment.py task=gaussian-linear method=pli method/model=gaussian
```
Here, we run a Gaussian model with PLI on the simple Gaussian-location task.
To keep track of the experimental settings, we use [hydra](https://hydra.cc).

For a comprehensive list of all available configuration parameters, 
simply run the script with a `--help` argument

```
python scripts/run_sbi_experiment.py --help

== Configuration groups ==
Compose your configuration from those groups (group=option)

method: abc, pli, snpe
method/model: conditioned-nsf, gaussian, maf, nsf, pmc
method/pseudo_likelihood: exponential-kernel, uniform-kernel
method/statistical_distance: mmd, wasserstein
task: furuta, gaussian-linear, gaussian-mixture, sir, slcp
task/posterior_sampling: sbibm
```

In particular, change the following settings to repeat the experiments from the paper.

```
python scripts/run_sbi_experiment.py \ 
    task=<TASK> 
    task.n_train_data=<N> \
    method=<METHOD>
    method/model=<MODEL>
```

The choice of the statistical distance can be set with `method/statistical_distance=<SD>`. 
The remaining hyperparameter settings are already set.
As an example, to run **PLI** with **MMD** on the **SLCP** task for **N=100** reference observations, run

```
python scripts/run_sbi_experiment.py task=slcp \
    task.n_train_data=100 \
    method=pli \
    method/model=nsf
    method/statistical_distance=mmd
```

In addition, we provide a script to create posterior samples 
that has been used to produce the reference posterior samples 
when the posterior is not analytically available.

```
python scripts/create_posterior_samples.py task=<TASK> task.n_train_data=<N>
```

Although, we provide posterior samples in `data`, this script can be used to generate data for different seeds.

### Logging
The experiments are logged with [wandb](https://wandb.ai). 
Before running an initial experiment, run `wandb login` and validate your profile 
with the authorization key provided from the website. 
You can switch off logging with wandb in `config.yaml` by toggling the saving flag.

### Citing
If you find the work useful, please consider citing the paper:

```
@inproceedings{gruner2023pli
  author =		 "Gruner, T. and  Belousov, B. and  Muratore, F. and  Palenicek, D. and  Peters, J.",
  year =		 "2023",
  title =		 "Pseudo-Likelihood Inference",
  booktitle =		 "Advances in Neural Information Processing Systems (NIPS / NeurIPS)",
}
```

### Acknowledgement

Our density estimators are based on [distrax](https://github.com/deepmind/distrax). 
We provide a [fork](https://github.com/theogruner/distrax) of distrax that includes support for conditional density estimators.