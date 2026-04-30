# hza_tagger

Jet tagger for H→Z(ll)+a(had) decays in CMS, targeting AK4 PUPPI jets from the hadronic decay of the pseudoscalar **a** (PDG 36).

```
hza_tagger/
├── common/          shared label defs, truth matching, IO schema, variable lists
├── converter/       btvNanoAOD ROOT → H5 (coffea, columnar)
├── tagger/          SALT submodule + training configs and scripts
└── analysis/        ROC curves, score distributions, working-point studies
```

## Quick start

### 1. Environment

```bash
mamba env create -f environment.yml
conda activate hza_tagger
pip install -e .        # installs common/, converter/, analysis/ as a package
```

### 2. SALT submodule

```bash
bash tagger/scripts/setup_salt.sh
```

### 3. Check your ROOT file's branch names

```bash
python converter/inspect_branches.py /path/to/hzanano_output_1.root
```

Compare with `common/variables.py` and adjust branch names there if needed.

### 4. Edit the converter config

```bash
vim converter/configs/hza_signal.yaml   # set file paths, cuts, chunk size
```

### 5. Run converter (local, quick test)

```bash
python converter/run_local.py \
    --config converter/configs/hza_signal.yaml \
    --out data/test_out.h5 \
    --max-events 5000
```

### 6. Scale out on DESY NAF / HTCondor

```bash
python converter/run_condor.py \
    --config converter/configs/hza_signal.yaml \
    --outdir data/chunks/ \
    --merge
```

### 7. Preprocess + train

```bash
bash tagger/scripts/preprocess.sh   # computes normalisation dict
bash tagger/scripts/train.sh        # launches SALT training
```

On DESY NAF GPU nodes, add `--trainer.accelerator gpu --trainer.devices 1` to `train.sh`.

### 8. Evaluate

```bash
# Score the test H5 with the best checkpoint
python analysis/scripts/eval_to_h5.py \
    --input  data/test.h5 \
    --ckpt   logs/hza_tagger*/checkpoints/best.ckpt \
    --config tagger/configs/hza_train.yaml \
    --output data/test_scores.h5

# Produce plots
python analysis/scripts/plots.py \
    --scores data/test_scores.h5 \
    --outdir analysis/plots/
```

## Tests

```bash
pytest -v
```

## Key design decisions

| Choice | Rationale |
|--------|-----------|
| Binary label (a-jet vs other) | Simplest discriminant; background jets taken from same signal sample |
| dR matching to a + daughters | Robust to multiple a's; requires all hadronic daughters inside the jet cone → clean merged-topology label |
| AK4 PUPPI jets | Standard CMS Run3 jet collection |
| PFCands as tracks | Rich per-constituent info in btvNanoAOD; IP variables zero-padded when absent (pheno files) |
| SALT via git submodule | No code fork; thin config layer only; easy to track upstream changes |
| Coffea columnar converter | Scales from laptop (iterative) to HTCondor (dask-jobqueue) without code changes |
