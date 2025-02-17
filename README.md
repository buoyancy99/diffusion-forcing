# Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion

#### [[Project Website]](https://boyuan.space/diffusion-forcing) [[Paper]](https://arxiv.org/abs/2407.01392)

[Boyuan Chen<sup>1</sup>](https://boyuan.space/), [Diego Martí Monsó<sup>2</sup>](https://www.linkedin.com/in/diego-marti/?originalSubdomain=de), [ Yilun Du<sup>1</sup>](https://yilundu.github.io/), [Max Simchowitz<sup>1</sup>](https://msimchowitz.github.io/), [Russ Tedrake<sup>1</sup>](https://groups.csail.mit.edu/locomotion/russt.html), [Vincent Sitzmann<sup>1</sup>](https://www.vincentsitzmann.com/) <br/>
<sup>1</sup>MIT <sup>2</sup>Technical University of Munich </br>

This is the v1.5 code base for our paper [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](https://boyuan.space/diffusion-forcing). The **main** branch contains our latest reimplementation with temporal attention (recommended) while the **paper** branch contains RNN code used by original paper for reproduction purpose. 

🔥 New: [Diffusion Forcing v2](https://boyuan.space/history-guidance/) is released! It is a stronger technique to roll out extremely long video generation, with modern architectures like DiT and latent diffusion. Please check out its [github repo](https://github.com/kwsong0113/diffusion-forcing-transformer) as well if you are only interested in video generation.

![plot](teaser.png)

```
@article{chen2025diffusion,
  title={Diffusion forcing: Next-token prediction meets full-sequence diffusion},
  author={Chen, Boyuan and Mart{\'\i} Mons{\'o}, Diego and Du, Yilun and Simchowitz, Max and Tedrake, Russ and Sitzmann, Vincent},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={24081--24125},
  year={2025}
}
```

# Project Instructions

## Setup

If you want to use our latest improved implementation for video and planning with temporal attention instead of RNN, stay on this branch. If you are instead interested in reproducing claims by orignal paper, switch to the branch used by original paper via `git checkout paper`.

Run `conda create python=3.10 -n diffusion-forcing` to create environment.
Run `conda activate diffusion-forcing` to activate this environment.

Install dependencies for time series, video and robotics:

```
pip install -r requirements.txt
```

[Sign up](https://wandb.ai/site) a wandb account for cloud logging and checkpointing. In command line, run `wandb login` to login.

Then modify the wandb entity in `configurations/config.yaml` to your wandb account.

Optionally, if you want to do maze planning, install the following complicated dependencies due to outdated dependencies of d4rl. This involves first installing mujoco 210 and then run

```
pip install -r extra_requirements.txt
```

## Quick start with pretrained checkpoints

Since dataset is huge, we provide a mini subset and pre-trained checkpoints for you to quickly test out our model! To do so, download mini dataset and checkpoints from [here](https://drive.google.com/file/d/1xAOQxWcLzcFyD4zc0_rC9jGXe_uaHb7b/view?usp=sharing) to project root and extract with `tar -xzvf quickstart_atten.tar.gz`. Files shall appear in `data` and `outputs/xxx.ckpt`. Make sure you also git pull upstream to use latest version of code if you forked before ckpt release!

Then run the following commands and go to the wandb panel to see the results.

### Video Prediction:

Our visualization is side by side, with prediction on the left and ground truth on the right. However, ground truth is expected to not align with prediction since the sequence is highly stochastic. Ground truth is provided to provide an idea about quality only.

Autoregressively generate minecraft video with 1x the length it's trained on:
`python -m main +name=sample_minecraft_pretrained load=outputs/minecraft.ckpt experiment.tasks=[validation]`

To let the model roll out **longer than it's trained on**, simply append `dataset.validation_multiplier=8` to the above commands, and it will rollout `8x` longer than maximum sequence length it's trained on.

The above checkpoint is trained for 100K steps with small number of frames. We've already verified diffusion forcing works in latent diffusion setting and can be extended to many more tokens without sacrificing compositionally (with some addition techniques outside this repo)! Stay tuned for our next project!

### Maze Planning:

The maze planning setting is changed a bit as we gain more insighs, please see corresponding paragraphs in training section for details. We haven't reimplemented MCTG yet, but you can already see nice visualizations on wandb log.

Medium Maze

`python -m main experiment=exp_planning algorithm=df_planning dataset=maze2d_medium dataset.action_mean=[] dataset.action_std=[] dataset.observation_mean=[3.5092521,3.4765592] dataset.observation_std=[1.3371079,1.52102] load=outputs/maze2d_medium_x.ckpt experiment.tasks=[validation] algorithm.guidance_scale=3 +name=maze2d_medium_x_sampling`

Large Maze

`python -m main experiment=exp_planning algorithm=df_planning dataset=maze2d_large dataset.observation_mean=[3.7296331,5.3047247] dataset.observation_std=[1.8070312,2.5687592] dataset.action_mean=[] dataset.action_std=[] load=outputs/maze2d_large_x.ckpt experiment.tasks=[validation] algorithm.guidance_scale=2 +name=maze2d_large_x_sampling`

We also explored a couple more settings but haven't reimplemented everything in original paper yet. If you are interestted in those checkpoints, see the source code of this README file for ckpt loading instructions that's commented out.

<!--
Here is also a position + velocity setting ckpt, but we don't recommend this because diffusing quantity and its derivative together creates some bad optimization landscape.

`python -m main experiment=exp_planning algorithm=df_planning dataset=maze2d_medium dataset.observation_std=[2.6742158,3.04204,9.3630628,9.4774808] dataset.action_mean=[] dataset.action_std=[] load=outputs/maze2d_medium_xv.ckpt experiment.tasks=[validation] algorithm.guidance_scale=4 +name=maze2d_medium_xv_sampling`

`python -m main experiment=exp_planning algorithm=df_planning dataset=maze2d_large dataset.observation_std=[3.6140624,5.1375184,9.747382,10.5974788] dataset.action_mean=[] dataset.action_std=[] load=outputs/maze2d_large_xv.ckpt experiment.tasks=[validation] algorithm.guidance_scale=4 +name=maze2d_large_xv_sampling`

Here is also ckpt where we take diffused actions,a challenging setting that's not done in prior papers. We haven't got it working as well as original RNN version of diffusion forcing, but it does have okay numbers. You can tune up the guidance scale a bit.

`python -m main experiment=exp_planning algorithm=df_planning dataset=maze2d_medium dataset.observation_std=[2.67,3.04,8,8] dataset.action_std=[6,6] load=outputs/maze2d_medium_xva.ckpt experiment.tasks=[validation] algorithm.guidance_scale=2 algorithm.open_loop_horizon=10 +name=maze2d_medium_xva_sampling`

`python -m main experiment=exp_planning algorithm=df_planning dataset=maze2d_large dataset.observation_std=[3.62,5.14,9.76,10.6] dataset.action_std=[3,3] load=outputs/maze2d_large_xva.ckpt experiment.tasks=[validation] algorithm.guidance_scale=2 algorithm.open_loop_horizon=10 +name=maze2d_large_xva_sampling` -->

## Training

### Video

Video prediction requires downloading giant datasets. First, if you downloaded the mini subset following `Quick start with pretrained checkpoints` section, delete the mini subset folders `data/minecraft` and `data/dmlab` because we have to download the whole dataset this time. We've coded in python that it will download the dataset for you it doesn't already exist. Due to the slowness of the [source](https://github.com/wilson1yan/teco), this may take a couple days. If you prefer to do it yourself via bash script, please refer to the bash scripts in original [TECO dataset](https://github.com/wilson1yan/teco) and use `dmlab.sh` and `minecraft.sh` in their Dataset section of README, any maybe split bash script into parallel scripts.

Then just run the corresponding commands:

#### Minecraft

`python -m main +name=your_experiment_name algorithm=df_video dataset=video_minecraft`

#### DMLab

`python -m main +name=your_experiment_name algorithm=df_video dataset=video_dmlab algorithm.weight_decay=1e-3 algorithm.diffusion.architecture.network_size=48 algorithm.diffusion.architecture.attn_dim_head=32 algorithm.diffusion.architecture.attn_resolutions=[8,16,32,64] algorithm.diffusion.beta_schedule=cosine`

#### No causal masking

Simply append `algorithm.causal=False` to your command.

#### Play with sampling

Please take a look at "Load a checkpoint to eval" paragraph to understand how to use load checkpoint with `load=`. Then, run the exact training command with `experiment.tasks=[validation] load={wandb_run_id}` to load a checkpoint and experiment with sampling.

To see how you can roll out longer than the sequence is trained on, you can find instructions in `quick start with pretrained checkpoints` section. Keep in mind that rolling out infinitely without sliding window is a property of original RNN implementation on `paper` branch, and this version has to use sliding window since it's temporal attention.

By default, we run autoregressive sampling with stablization. To sample next 2 tokens jointly, you can append the following to the above command: `algorithm.scheduling_matrix=full_sequence algorithm.chunk_size=2`.

## Maze Planning

For those who only wish to reproduce the original paper instead of transformer architecture, please checkout`paper` branch of the code instead.

**Medium Maze**

`python -m main experiment=exp_planning algorithm=df_planning dataset=maze2d_medium dataset.action_mean=[] dataset.action_std=[] dataset.observation_mean=[3.5092521,3.4765592] dataset.observation_std=[1.3371079,1.52102] +name=maze2d_medium_x`

**Large Maze**

`python -m main experiment=exp_planning algorithm=df_planning dataset=maze2d_large dataset.observation_mean=[3.7296331,5.3047247] dataset.observation_std=[1.8070312,2.5687592] dataset.action_mean=[] dataset.action_std=[] +name=maze2d_large_x`

**Run planning after model is trained**

Please take a look at "Load a checkpoint to eval" paragraph to understand how to use load checkpoint with `load=`. To sample, simply append `load={wandb_id_of_above_runs} experiment.tasks=[validation] algorithm.guidance_scale=2 +name=maze2d_sampling` to above command after trained. Feel free to tune the `guidance_scale` from 1 - 5.

This version of maze planning uses a different version of diffusion forcing from original paper - while doing the follow up to diffusion forcing, we realized that training with independent noise actually constructed a smooth interpolation between causal and non-causal models too, since we can just masked out future by complete noise (fully causal) or some noise (interpolation). The best thing is, you can still account for causal uncertainty via pyramoid sampling in this setting, by masking out tokens at different noise levels, and you can still have flexible horizon because you can tell the model that padded entries are pure noise, a unique ability of diffusion forcing.

We also reflected a bit about the environment and concluded that the original metric isn't necessarily a good metric, because maze planning should reward those who can plan the fastest route to goal, not a slow walking agent that goes there at the end of episode. The dataset never contains data of staying at the goal, so agents are supposed to walk away after reaching the goal. I think [Diffuser](https://arxiv.org/abs/2205.09991) had an unfair advantage of just generating slow plans, that happend to let the agent stay in the neighbour hood of goal for longer and got very high reward, exploiting flaws in the environment design (a good design would involve penalty of longer time taken to reach goal). So, in this version of code, we just optimize for flexible horizon planning that tries to reach goal asap, and the planner will automatically come back to goal if it left the goal since staying is never in dataset. You can see new metrics we designed in wandb logging interface.

## Timeseries and Robotics

Please checkout `paper` branch for the code used by original paper. If I have time later, I will reimplement these two domains with transformer as well to complete this branch.

# Change Log

| Data      |                                              Notes                                              |
| --------- | :---------------------------------------------------------------------------------------------: |
| Jul/30/24 |             Upgrade RNN to temporal attention, move orignal code to 'paper' branch              |
| Jul/03/24 | Initial release of the code. Email me if you have questions or find any errors in this version. |

# Infra instructions

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in `README.md` and the `LICENSE` file to credit the author.

All experiments can be launched via `python -m main +name=xxxx {options}` where you can fine more details later in this article.

The code base will automatically use cuda or your Macbook M1 GPU when available.

For slurm clusters e.g. mit supercloud, you can run `python -m main cluster=mit_supercloud {options}` on login node.
It will automatically generate slurm scripts and run them for you on a compute node. Even if compute nodes are offline,
the script will still automatically sync wandb logging to cloud with <1min latency. It's also easy to add your own slurm
by following the `Add slurm clusters` section.

## Modify for your own project

First, create a new repository with this template. Make sure the new repository has the name you want to use for wandb
logging.

Add your method and baselines in `algorithms` following the `algorithms/README.md` as well as the example code in
`algorithms/diffusion_forcing/df_video.py`. For pytorch experiments, write your algorithm as a [pytorch lightning](https://github.com/Lightning-AI/lightning)
`pl.LightningModule` which has extensive
[documentation](https://lightning.ai/docs/pytorch/stable/). For a quick start, read "Define a LightningModule" in this [link](https://lightning.ai/docs/pytorch/stable/starter/introduction.html). Finally, add a yaml config file to `configurations/algorithm` imitating that of `configurations/algorithm/df_video.yaml`, for each algorithm you added.

Add your dataset in `datasets` following the `datasets/README.md` as well as the example code in
`datasets/video`. Finally, add a yaml config file to `configurations/dataset` imitating that of
`configurations/dataset/video_dmlab.yaml`, for each dataset you added.

Add your experiment in `experiments` following the `experiments/README.md` or following the example code in
`experiments/exp_video.py`. Then register your experiment in `experiments/__init__.py`.
Finally, add a yaml config file to `configurations/experiment` imitating that of
`configurations/experiment/exp_video.yaml`, for each experiment you added.

Modify `configurations/config.yaml` to set `algorithm` to the yaml file you want to use in `configurations/algorithm`;
set `experiment` to the yaml file you want to use in `configurations/experiment`; set `dataset` to the yaml file you
want to use in `configurations/dataset`, or to `null` if no dataset is needed; Notice the fields should not contain the
`.yaml` suffix.

You are all set!

`cd` into your project root. Now you can launch your new experiment with `python main.py +name=<name_your_experiment>`. You can run baselines or
different datasets by add arguments like `algorithm=xxx` or `dataset=xxx`. You can also override any `yaml` configurations by following the next section.

One special note, if your want to define a new task for your experiment, (e.g. other than `training` and `test`) you can define it as a method in your experiment class and use `experiment.tasks=[task_name]` to run it. Let's say you have a `generate_dataset` task before the task `training` and you implemented it in experiment class, you can then run `python -m main +name xxxx experiment.tasks=[generate_dataset,training]` to execute it before training.

## Pass in arguments

We use [hydra](https://hydra.cc) instead of `argparse` to configure arguments at every code level. You can both write a static config in `configuration` folder or, at runtime,
[override part of yur static config](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) with command line arguments.

For example, arguments `algorithm=example_classifier experiment.lr=1e-3` will override the `lr` variable in `configurations/experiment/example_classifier.yaml`. The argument `wandb.mode` will override the `mode` under `wandb` namesspace in the file `configurations/config.yaml`.

All static config and runtime override will be logged to cloud automatically.

## Resume a checkpoint & logging

For machine learning experiments, all checkpoints and logs are logged to cloud automatically so you can resume them on another server. Simply append `resume={wandb_run_id}` to your command line arguments to resume it. The run_id can be founded in a url of a wandb run in wandb dashboard. By default, latest checkpoint in a run is stored indefinitely and earlier checkpoints in the run will be deleted after 5 days to save your storage.

On the other hand, sometimes you may want to start a new run with different run id but still load a prior ckpt. This can be done by setting the `load={wandb_run_id / ckpt path}` flag.

## Load a checkpoint to eval

The argument `experiment.tasks=[task_name1,task_name2]` (note the `[]` brackets here needed) allows to select a sequence of tasks to execute, such as `training`, `validation` and `test`. Therefore, for testing a machine learning ckpt, you may run `python -m main load={your_wandb_run_id} experiment.tasks=[test]`.

More generally, the task names are the corresponding method names of your experiment class. For `BaseLightningExperiment`, we already defined three methods `training`, `validation` and `test` for you, but you can also define your own tasks by creating methods to your experiment class under intended task names.

## Debug

We provide a useful debug flag which you can enable by `python main.py debug=True`. This will enable numerical error tracking as well as setting `cfg.debug` to `True` for your experiments, algorithms and datasets class. However, this debug flag will make ML code very slow as it automatically tracks all parameter / gradients!

## Add slurm clusters

It's very easy to add your own slurm clusters via adding a yaml file in `configurations/cluster`. You can take a look
at `configurations/cluster/mit_vision.yaml` for example.
