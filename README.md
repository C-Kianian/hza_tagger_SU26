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
├── common/             shared label defs, truth matching, IO schema, variable lists
├── converter/          btvNanoAOD ROOT → H5 (coffea, columnar)
├── tagger/             SALT submodule + training configs and scripts
├── atlas_2025_model/   Replica of ATLAS 2025 model in Salt (see literature)
├── analysis/           plotting scripts for ROC curves, score distributions
└── warmup_material/    "original" provides starter code, "my version" is a (roughly) complete guide
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

<details>
<summary>Helpful, but not required, tip!</summary>

After running steps 1 and 2 once they can be simplified into a single command to save time. It is best to ask someone or AI for your specific setup. Here is my `~/.bashrc` example:

```bash
# Shortcut to jump straight into the tagger project
go_tagger() {
    # 1. Navigate to your project repository (Replace with your actual absolute path!)
    cd /path/to/hza_tagger_SU26/ || return

    # 2. Activate the environment
    conda activate hza_tagger

    # 3. Run the salt setup script
    bash tagger/scripts/setup_salt.sh

    echo "🚀 hza_tagger env and Salt setup are ready!"
}
```
Then everytime I log in I simply run:
```bash
go_tagger
```
And after ~1 min everything is set up
</details>

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
For larger files it is needed to split and run them in parallel to save time: 
```bash
python converter/run_condor.py \
    --config converter/configs/hza_signal.yaml \
    --outdir data/chunks/ \
    --merge
```

Importantly, the condor script only produces one merged file, it is necessary to split this into train/test/val, this can be done via:

```bash
python converter/make_train_test_val.py \ # check the script to see all args
        --inputs file1.h5 file2.h5 etc.h5 \
        -outdir=/path/to/out/dir \
        --name=name_to_add_to_files
```
Note this script takes in multiple files, so one can merge a background file with a signal file and then split. Importantly, if the input files are reordered the split will not be consistent.  

<details>
<summary>Cyrus' Work</summary>

Most of my work can be found in:

`/data/dust/user/kianianc/`

```text
/data/dust/user/kianianc/
├── H_Za_data/          Contains h5 file data
│   ├── 1st_run         The original h5 processing run
│   ├── 2nd_run         Adds eta, phi track info for edge features
│   ├── 3rd_run         Adds ATLAS features, filter, and truth-failing daughter info
│   └── test_sig        Test signal files guaranteed to work
├── model_logs/         Includes logs and checkpoints for trained models
├── plots/              Plots for the models and data
└── split_300k_bkg/     Background file split into smaller ROOT files
```

The different versions of models are detailed [here](https://docs.google.com/presentation/d/1hy5rmuNpHurtNS1Z8F4DmpsISaKPKuabucHzqMwENwk/edit?usp=sharing)

</details>

### 5.1. Validate h5 Files
To ensure the h5 processing was successful run:
```bash
python analysis/scripts/data_validation_scripts/sig_vs_bkg_plots.py 
--file=/path/to/your/h5 \
--outdir=/path/to/data/plots \
--plot 
```

Additionally, one can include `--atlas` to plot ATLAS paper variables; `--edg` for `Salt` edge features; or set a max number of events to plot with `--maxEvents=N`

Also included is `analysis/scripts/data_validation_scripts/sig_vs_sig_plots.py`, which plots different h5 files against each other.

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

### 6.1. Preprocess + train (ATLAS version)
```bash
python atlas_2025_model/scripts/event_mask.py 
  --infile /path/to/data \
  --outdir /path/to/dir \ 
  --mask sample_name \    # applies ATLAS selection criteria (ie. atlas_valid)
  # This can also be used to filter files into only signal/background events
  
bash tagger/scripts/preprocess.sh /path/to/your/train/file # computes normalisation dict

bash tagger/scripts/train.sh 
  /path/to/your/train/file \
  /path/to/your/val/file \
  /path/to/your/test/file  # launches SALT training
```

### 7. Evaluate

The evaluation script auto-discovers the test H5 file, the most recent checkpoint, and the training config from the standard project layout:

```bash
bash analysis/scripts/evaluate.sh --config=/path/to/train_cfg --plot 
```

Note: This script works for ATLAS models too

It runs two steps in sequence and writes plots to `analysis/plots_in_file_name/`

1. **Score** — `eval_to_h5.py` loads the best checkpoint and appends a `scores` dataset (shape `(N, 2)`) to a copy of the test H5.
2. **Plot** — `plots.py` produces ROC curves, score distributions, and efficiency vs pT/η.

**Override any path** (best practice) via argument or environment variable:

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
One can also specifiy `--modeldir=/path/to/logs`, the path to the directory that is the parent of `ckpts`, to make use of the best model finding feature rather than having to explicity pass it themselves


### 7.1. Additional Analysis
Additional analysis scripts, under `analysis/scripts`, that are not already run by default in `evaluate.sh` include:
1. `mass_specific_classifier_plots.py` which plots the different masses and ATLAS paper masses for a classifier trained on all mass points
2. `overlay_plots.py` which currently adds different ROC curves to the same plot 


## Tests

```bash
pytest -v
```

