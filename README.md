## Generalized Position-Bias Model (GPBM)

This repo contains the code for the experiments in our paper "Generalized Position-Based Model: Rethinking Position Weights in Ranking Off-Policy Evaluation" published in RecSys 2026.

### Citation

If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to our RecSys 2026 paper:
```
@inproceedings{knyazev2026Generalized,
  Author = {Knyazev, Norman and Bellini, Vito and Yurtseven, Huseyin and London, Ben},
  Title = {Generalized Position-Based Model: Rethinking Position Weights in Ranking Off-Policy Evaluation},
  Booktitle={Proceedings of the 20th ACM Conference on Recommender Systems},
  Publisher = {Association for Computing Machinery}
  Year = {2026}
}
```

We perform off-policy evaluation (OPE) for ranking, comparing our new estimator GPBM against baseline estimators (IPM, PBM, INTERPOL, etc.) under varying position bias, policy temperatures, and sample sizes.
Our code is split into two high-level parts. Our first part (the pipeline) is used to train logging and target policies, calculate propensities, generate user clicks, optimize GPBM's $F$-matrix and evaluate GPBM and the baselines across different settings.
The second part, run via a single jupyter notebook, is used to process and visualize the outputs of the pipeline.

### Requirements

- Python 3.10+
- The code has been tested on a CUDA-compatible GPU with 16 GB VRAM alongside an 8-core CPU with 16GB RAM. The code also runs fully on CPU but with GPU it will be significantly faster.
- GNU `parallel` (optional, for multi-GPU parallelization): `sudo apt install parallel`

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `torch`, `tensorflow`, `tfds-nightly`, `tqdm`, `cvxpy`, `jupyterlab`.
Result visualization also assumes a local installation of latex. 

### Dataset Setup

Follow the instructions in `./dataset_instructions.md` to download Yahoo and MSLR datasets. 

### Project Structure

```
ope/
├── generate_configs.py       # Generate experiment configs from a YAML metaconfig
├── run_pipeline.sh           # Run the full pipeline (steps 0-4)
├── training/                 # Model training (step 0)
├── get_clicks_propensities.py # Click simulation & propensity computation (steps 1-2)
├── ipm.py                    # Non-PBM user click simulation (step 1)
├── optimize_F.py             # F-matrix optimization for GPBM (step 3)
├── evaluate.py               # OPE evaluation & MSE computation (step 4)
├── gpbm.py                   # GPBM estimator
├── ensemble.py               # Ensemble methods (OPERA, BLUE, SLOPE)
├── PL/                       # Plackett-Luce sampling & propensities
├── data.py                   # Dataset loading & preprocessing
└── utils.py                  # Shared utilities
```

### Config Generation

Each pipeline step requires a config specifying the dataset, fold, position bias type, and step-specific parameters (temperatures, sample counts, bias misestimation, etc.). Configs are generated in bulk from a YAML metaconfig:

```bash
python3 ope/generate_configs.py configs/experiments/yahoo_pbm.yaml
```

The metaconfig defines:
- `base_config_path`: where generated configs are written (e.g., `~/ope/configs/runs/<exp_name>`)
- `base_exp_path`: where experiment outputs (checkpoints, clicks, results) are stored
- Dataset, folds, policy temperatures, position bias types, bias misestimation patterns, etc.

See `configs/experiments/yahoo_pbm.yaml` for the full configuration. Edit the paths at the top to match your machine.

All example paths below assume `base_config_path=~/ope/configs/runs` and `exp_name=yahoo_pbm`.
To run the experiments on MSLR, simply replace `yahoo` with `mslr` in the remainder of this README.
To run the experiments using non-PBM user clicks, run Step 0 as-is and for remaining steps replace `pbm` with `ipm`.

### Pipeline Steps

The pipeline has 5 steps (0-4). Each can be run individually or together via `run_pipeline.sh`.

#### Step 0: Train ranking policies

Train logging and target policies on true document relevance labels. The same trained model is reused across all subsequent steps for a given dataset+fold.

```bash
# Logging policy (uses a subset of features)
python3 -m ope.training.train ~/ope/configs/runs/yahoo_pbm/training/yahoo/fold1/2_3_worst_features.py

# Target policy (uses all features)
python3 -m ope.training.train ~/ope/configs/runs/yahoo_pbm/training/yahoo/fold1/all_features.py
```

#### Step 1: Generate logging policy clicks and propensities

Compute Plackett-Luce propensities for the logging policy, then sample rankings and simulate user clicks (50 repeats). The (inverse) temperature parameter controls policy determinism. Clicks are determined by document relevance and the true position bias curve.

```bash
python3 -m ope.get_clicks_propensities ~/ope/configs/runs/yahoo_pbm/logging/yahoo/fold1/2_3_worst_features/pbdcg/ndocs50/temp0_6/config.py
```

#### Step 2: Generate target policy propensities

Compute propensities for the target policy (no click simulation needed).

```bash
python3 -m ope.get_clicks_propensities ~/ope/configs/runs/yahoo_pbm/logging/yahoo/fold1/all_features/pbdcg/ndocs50/temp1/config.py
```

Clicks and propensities from steps 1-2 are shared across steps 3-4 for configs that differ only in estimated position bias.

#### Step 3: Optimize the F-matrix

Optimize and save the GPBM F-matrix separately for each repeat's clicks. Key parameters: `position_bias` (estimated curve) and `eps_k` (uncertainty bound on the estimate).

```bash
python3 -m ope.optimize_F ~/ope/configs/runs/yahoo_pbm/F/yahoo/fold1/2_3_worst_features/all_features/nsampled5/pbdcg/ndocs50/temp0_6/temp1/bias_pow0_8/eps_times1_5/config.py
```

#### Step 4: Evaluate

Estimate the target policy value using GPBM and baseline estimators for each repeat, compute MSE, and output results as JSON.

```bash
python3 -m ope.evaluate ~/ope/configs/runs/yahoo_pbm/eval/yahoo/fold1/2_3_worst_features/all_features/nsampled5/pbdcg/ndocs50/temp0_6/temp1/bias_pow0_8/eps_times1_5/config.py
```

### Running the Full Pipeline

`run_pipeline.sh` runs all steps for specified parameter combinations. Multiple values can be passed per parameter (space-separated in quotes):

```bash
ope/run_pipeline.sh \
  --exp_name yahoo_pbm \
  --dataset yahoo \
  --logging_model 2_3_worst_features \
  --target_model all_features \
  --logging_temp 0.6 \
  --target_temp 1 \
  --pb pbdcg \
  --ndocs 50 \
  --nsampled "1" \
  --bias_pattern "bias_pow0_8 bias_minus0_1" \
  --eps_pattern "eps_times1_5" \
  --steps 01234 \
  --folds "1 2" \
  --config_base ~/ope/configs/runs
```

Use `--steps` to run only specific steps (e.g., `--steps 34` for optimization and evaluation only).

In our experiments we use the following bias_patterns:
`bias_pow1_4 bias_pow1_2 bias_plus0 bias_pow0_8 bias_pow0_6 bias_plus0_1 bias_minus0_1 bias_zigzag0_1`
and eps_k patterns:
`eps_times0 eps_times0_25 eps_times0_5 eps_times1 eps_times2 eps_times3 eps_times1_5 eps_times0_5_to_1_5 eps_times1_5_to_0_5`


### Figure Generation
Finally, use `ope/notebooks/figures.ipynb` to visualize the outputs of the above pipeline.

### Bonus: Multi-GPU Pipeline Parallelization

With multiple GPUs, use GNU `parallel` to distribute pipeline runs. Split into phases to avoid redundant computation (training and click generation are shared across downstream configs):

```bash
export exp_name="{9}_pbm"  # {9} is replaced by the dataset name
N=$(nvidia-smi -L | wc -l)
JOBS=$((2 * N))  # Two runs per GPU (adjust based on VRAM)

# Phase 1: Train policies
parallel --jobs $JOBS --bar --results outdir --joblog jobs.log \
  'OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCH_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$(( ({%} - 1) % '"$N"' )) ope/run_pipeline.sh --exp_name '"$exp_name"' --dataset {9} --logging_model 2_3_worst_features --target_model all_features --logging_temp {2} --target_temp {3} --pb {4} --ndocs {5} --nsampled {6} --bias_pattern {7} --eps_pattern {8} --steps 0 --folds {1} --config_base ~/ope/configs/runs' \
  ::: 1 2 \
  ::: 1 \
  ::: 1 \
  ::: pbdcg \
  ::: 50 \
  ::: 1 \
  ::: bias_pow0_8 \
  ::: eps_times1 \
  ::: yahoo

# Phase 2: Generate clicks and propensities
parallel --jobs $JOBS --bar --results outdir --joblog jobs.log \
  'OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCH_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$(( ({%} - 1) % '"$N"' )) ope/run_pipeline.sh --exp_name '"$exp_name"' --dataset {9} --logging_model 2_3_worst_features --target_model all_features --logging_temp {2} --target_temp {3} --pb {4} --ndocs {5} --nsampled {6} --bias_pattern {7} --eps_pattern {8} --steps 12 --folds {1} --config_base ~/ope/configs/runs' \
  ::: 1 2 \
  ::: 0.6 \
  ::: 1 \
  ::: pbdcg \
  ::: 50 \
  ::: 1 \
  ::: bias_pow0_8 \
  ::: eps_times1 \
  ::: yahoo

# Phase 3: Optimize F and evaluate
parallel --jobs $JOBS --bar --results outdir --joblog jobs.log \
  'OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCH_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$(( ({%} - 1) % '"$N"' )) ope/run_pipeline.sh --exp_name '"$exp_name"' --dataset {9} --logging_model 2_3_worst_features --target_model all_features --logging_temp {2} --target_temp {3} --pb {4} --ndocs {5} --nsampled {6} --bias_pattern {7} --eps_pattern {8} --steps 34 --folds {1} --config_base ~/ope/configs/runs' \
  ::: 1 2 \
  ::: 0.6 \
  ::: 1 \
  ::: pbdcg \
  ::: 50 \
  ::: 1 5 \
  ::: bias_pow0_8 bias_minus0_1 \
  ::: eps_times1 \
  ::: yahoo
```

The three-phase split ensures policies are trained once (phase 1), clicks are generated once per temperature/bias combination (phase 2), and only the F optimization and evaluation (phase 3) fan out across all parameter combinations.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under Creative Commons Attribution-NonCommercial 4.0 International license.
