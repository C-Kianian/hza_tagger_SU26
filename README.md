# Jet tagging of low mass resonances

This project provides tools to train and evaluate a jet tagger for a H→Z(ll)+a(had) search at the CMS experiment.
It targets AK4 PUPPI jets from the hadronic decay of the pseudoscalar **a** (PDG 36).
The algorithm learns to distinguish jets that originate from the hadronic decay of the **a** from other QCD jets using the kinematic properties of the jets and resolving the jet substructure.
This is achieved by storing both the jet and the associated PFCandidates matched to the jet.
The training makes use of per-jet labels from MC truth simulation as **a** jet or background jet.

## Description of project

The project has several sub-parts which are explained below:

```
hza_tagger/
├── common/          shared label defs, truth matching, IO schema, variable lists
├── converter/       btvNanoAOD ROOT → H5 (coffea, columnar)
├── tagger/          SALT submodule + training configs and scripts
└── analysis/        plotting scripts for ROC curves, score distributions
```

1. As the first step, you will have to prepare the datasets in the [h5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) format which can be used to train the machine learning algorithm. This is handled by the tools in `converter`.

2. The second step is the training of the algorithm. This is handled by the tools inside `tagger`, which make use of the [`salt`](https://ftag-salt.docs.cern.ch) software.

3. The third step is the evaluation of the tagger performance on test datasets created in the first step using the scripts in `analysis`.

## Literature

- Barr et al., (2025). Salt: Multimodal Multitask Machine Learning for High Energy Physics. Journal of Open Source Software, 10(112), 7217, https://doi.org/10.21105/joss.07217
- Chisholm, A.S., Kuttimalai, S., Nikolopoulos, K. et al. Measuring rare and exclusive Higgs boson decays into light resonances. Eur. Phys. J. C 76, 501 (2016). https://doi.org/10.1140/epjc/s10052-016-4345-9
- ATLAS Collaboration. Search for Higgs Boson Decays into a 𝑍 Boson and a Light Hadronically Decaying Resonance Using 13 TeV 𝑝⁢𝑝 Collision Data from the ATLAS Detector. Phys. Rev. Lett. 125, 221802 – Published 25 November, 2020. https://doi.org/10.1103/PhysRevLett.125.221802
- ATLAS Collaboration. Search for Higgs boson decays into a Z boson and a light hadronically decaying resonance in pp collisions at 13 TeV with the ATLAS detector. Physics Letters B Volume 868, September 2025, 139671. https://doi.org/10.1016/j.physletb.2025.139671


## Quick start

### 1. Environment

```bash
mamba env create -f environment.yml
conda activate hza_tagger
pip install -e .        # installs common/, converter/, analysis/ as a package, the "." is important here!
```

### 2. SALT submodule

```bash
bash tagger/scripts/setup_salt.sh
```

### 3. Prepare the converter config

Check your input ROOT file's branch names

```bash
python converter/inspect_branches.py /path/to/hzanano_output_1.root
```

Compare with `common/variables.py` and adjust branch names there if needed.


```bash
nano converter/configs/hza_signal.yaml   # set file paths, cuts, chunk size
```

### 4. Run converter (local, quick test)

```bash
python converter/run_local.py --config converter/configs/hza_signal.yaml
```

This reads `split_fractions` from the config (default 70 / 15 / 15 %) and writes
three files: `data/train.h5`, `data/val.h5`, `data/test.h5`.

Pass `--out data/all.h5` to skip the split and write a single file (useful for quick tests).
Pass `--max-events N` to cap the number of events read.

### 5. Scale out on DESY NAF / HTCondor

```bash
python converter/run_condor.py \
    --config converter/configs/hza_signal.yaml \
    --outdir data/chunks/ \
    --merge
```

### 6. Preprocess + train

This makes most sense to run on a GPU machine. If you run this on your local computer, you will not have a good time. You can run to test it, but it will be very slow. It is better to move to DESY NAF with GPU access.

Open this page and read it please: [https://docs.desy.de/naf/documentation/gpu-on-naf/](https://docs.desy.de/naf/documentation/gpu-on-naf/)

```bash
bash tagger/scripts/preprocess.sh   # computes normalisation dict
bash tagger/scripts/train.sh        # launches SALT training
```

On DESY NAF GPU nodes, add `--trainer.accelerator gpu --trainer.devices 1` to `train.sh`.

**Comet.ml logging** is enabled automatically when a `COMET_API_KEY` is present.

<details>
<summary>Setting up a Comet account (first time)</summary>

1. Go to **[comet.com](https://www.comet.com)** and sign up for a free account.
2. After logging in, open **[comet.com/api/my/settings](https://www.comet.com/api/my/settings)** and copy your **API key**.
3. Create `.env` in the project root (it is git-ignored):
   ```bash
   touch .env && nano .env
   # then open .env and paste your key:
   #   COMET_API_KEY=<your_key>
   ```

</details>

`train.sh` sources `.env` on every run and passes the key to `CometLogger`. Without a key it falls back to offline mode (logs saved under `logs/`).

### 7. Evaluate

The evaluation script auto-discovers the test H5 file, the most recent checkpoint, and the training config from the standard project layout:

```bash
bash analysis/scripts/evaluate.sh
```

It runs two steps in sequence and writes plots to `analysis/plots/`:

1. **Score** — `eval_to_h5.py` loads the best checkpoint and appends a `scores` dataset (shape `(N, 2)`) to a copy of the test H5.
2. **Plot** — `plots.py` produces ROC curves, score distributions, and efficiency vs pT/η.

**Override any path** via argument or environment variable:

```bash
# Explicit test file
bash analysis/scripts/evaluate.sh data/my_test.h5

# Explicit test file + checkpoint
bash analysis/scripts/evaluate.sh data/my_test.h5 logs/my_run/checkpoints/best.ckpt

# Environment variable overrides
TEST_FILE=data/my_test.h5 \
CKPT=logs/my_run/checkpoints/best.ckpt \
PLOT_DIR=analysis/plots/my_run \
bash analysis/scripts/evaluate.sh
```

## Tests

```bash
pytest -v
```
