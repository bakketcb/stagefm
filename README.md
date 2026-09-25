# A Multimodal Radiomics Foundation Model for Gastric Cancer Staging in Real-World Multi-Center Imaging Cohorts

This repository holds the model, data-pipeline, training and evaluation code for the
staging system described in the manuscript. The system composes a frozen abdominal-CT
vision-language encoder adapted with low-rank updates, a cross-attention fusion block
over the peritumoral radiomic descriptors and the structured clinical, endoscopic and
pathology streams, a stage-consistency block that restricts the joint label space to
combinations the staging definitions admit, and a site-conditional risk-control layer
that fixes the treatment decision boundary per site stratum.

The importable package is `stagefm`. The primary endpoint the code measures is
treatment-boundary discordance: the share of examinations on which the predicted stage
implies a different management category from the one surgical pathology supports.

## Continuous integration

`ruff`, `black`, `isort` and `mypy --strict` run over `src/`, and the test suite runs
in the pre-commit configuration shipped in the repository. They are invoked directly:

```bash
ruff check .
black --check --line-length 160 .
isort --check-only --profile black --line-length 160 .
mypy --strict src/stagefm
pytest -q
```

## Installation

The pinned runtime is Python 3.11 or newer with torch 2.11; `requirements.txt`
carries the exact versions the verification pass ran against.

pip:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

conda:

```bash
conda env create -f environment.yml
conda activate stagefm
pip install -e .
```

Container:

```bash
docker build -t stagefm .
docker run --rm stagefm
```

The image is pinned to a CUDA 12.4 runtime whose torch build matches
`requirements.txt`, because the reported training profile is one node with four
accelerators. On a host without an accelerator, replace the torch install line with
the CPU wheel index and everything except the throughput figures runs unchanged.

## Data

The multi-centre clinical cohort is not distributed. It is held under a data-usage
statement issued by the corresponding author, the manuscript's data-availability
statement says the same, and the records cannot be released at record level. Nothing
in this repository reconstructs, approximates or redistributes it.

Because of that, the shipped pipeline reads a cohort drawn from
`src/stagefm/data/synthetic.py`, which reproduces the manuscript's *cohort shape* --
the site counts, the layer sizes, the median nodal yield per site, the missingness
counts -- and, more importantly, the structural properties the analyses depend on:

* the pathological nodal label is a sampling outcome, so a site that examines fewer
  nodes records a lower and noisier nodal category for the same underlying burden;
* labels are drawn only from the achievable set, so the joint label space matches the
  staging definitions;
* the imaging latent carries the underlying burden rather than the recorded label,
  which is what makes the reference standard, rather than the representation, the
  limiting factor on the nodal axis.

Every value computed from that generator is a generated value and is reported as
`NOT_RUN`. No number in any table in this repository may be presented as a study
result. To run on real records, replace the generator call in
`stagefm.cli.pipeline.prepare` with a reader that produces `Examination` objects with
the same fields; the schema contract is in `src/stagefm/data/schema.py`.

Manifest fields expected per examination: `record_id`, `site`, `region`, `stage`,
`harvested_nodes`, `scanner_vendor`, `neoadjuvant_exposed`, `lauren`, `age`, `sex`,
`ct_ref`, `radiomics`, `clinical`, `imaging_latent`, `endoscopy_terms`,
`pathology_terms`, `available`.

Cohort contract as declared in `configs/data/cohort.yaml`:

| Layer | Records | Sites |
| --- | --- | --- |
| training | 5,868 | A, B, C |
| internal test | 1,960 | A, B, C |
| external | 3,456 | D, E |
| prospective (silent mode) | 1,850 | A-D |

Site median harvested node counts run from 16 (site E) to 32 (site A); the adequacy
cutoff is 16 nodes. Nineteen records of the 11,284-record analysis set carry no
endoscopic description and 61 carry no pathology text; those records are retained and
signalled by an absence token rather than dropped.

`scripts/prepare_data.sh` materialises the analysis set and writes its manifest:

```bash
scripts/prepare_data.sh
```

It writes `<cohort root>/manifest.json` plus one JSON file per layer, reports the
layer sizes, the site table and the node-stratum summary, and prints a payload digest
over the harvested-node field so a re-run can be compared byte for byte.

External resources, their access conditions and their licences are listed in
`dataset_urls.txt` and `THIRD_PARTY_NOTICES.txt`. Only the three public resources the
manuscript names are used, and none of them carries stage labels.

## Repository layout

```
configs/
  data/cohort.yaml          cohort contract plus imaging and descriptor settings
  model/default.yaml        encoder, adaptation, fusion, heads, risk layer, objective
  train/default.yaml        optimisation profile
  experiment/*.yaml         one file per reported arm, ablation and supplementary analysis
src/stagefm/
  data/                     schema, staging definitions, cohort, synthetic cohort,
                            imaging, localisation, descriptors, text, streams, dataset
  models/                   encoder, low-rank adapters, fusion, ordinal heads,
                            feasibility projection, risk control, arms
  losses/                   ordinal, decision and calibration terms
  metrics/                  discrimination, concordance, calibration, decision, feasibility
  stats/                    DeLong, bootstrap, McNemar, variance components, FDR
  evaluation/               inference loop, per-site, subgroups, ascertainment,
                            sensitivity, reader study, in vitro, silent mode, reporting
  training/                 optimizer, schedule, precision, checkpoints, EMA, trainer
  cli/                      pipeline assembly, train, eval, verify
tests/                      unit, metric, model, training and verification tests
scripts/                    data preparation and launcher examples
```

## Paper-to-code map

The full mapping, with one entry per claim and its paper location, is written to
`claim_to_code.json` by the verification pass. The load-bearing entries:

| Paper location | What it defines | Where it lives |
| --- | --- | --- |
| Methods Sec. 4.1 | treatment-boundary rule, label space, cohort layers | `data/staging.py`, `data/schema.py`, `data/cohort.py` |
| Methods Sec. 4.3 | 1 mm isotropic resampling, HU window, bias-field and intensity standardisation, automated localisation, 0-5 mm peritumoral shell, stability and redundancy filters, per-site z-scoring | `data/imaging.py`, `data/segmentation.py`, `data/radiomics.py` |
| Methods Sec. 4.5 | the four modules: frozen encoder with low-rank adaptation, cross-attention fusion, stage-consistency projection, site-conditional risk control | `models/encoder.py`, `models/lora.py`, `models/fusion.py`, `models/stage_consistency.py`, `models/risk_control.py`, `models/stagefm.py` |
| Algorithm 1 | stage-consistent training with feasibility projection | `models/stagefm.py`, `losses/`, `training/trainer.py` |
| Algorithm 2 | site-conditional risk control and its finite-sample bound | `models/risk_control.py` |
| Algorithm 3 | feasibility projection onto the achievable set | `models/stage_consistency.py` |
| Algorithm 4 | bootstrap variance-component decomposition | `stats/icc.py`, `stats/bootstrap.py` |
| Methods Sec. 4.7 | optimizer, schedule, rank, batch, grid, epochs, augmentations, five runs, compute | `configs/train/default.yaml`, `training/optim.py`, `training/scheduler.py`, `data/imaging.py` |
| Methods Sec. 4.8 | comparison arms and the quoted reference points | `models/baselines.py`, `configs/experiment/baseline_*.yaml` |
| Methods Sec. 4.9-4.10 | reader study and in vitro protocol | `evaluation/reader_study.py`, `evaluation/in_vitro.py` |
| Methods Sec. 4.11 | DeLong intervals, weighted kappa, calibration error, net benefit, reclassification, paired testing, variance decomposition, per-family FDR, missingness, sensitivity analyses | `stats/`, `metrics/`, `evaluation/sensitivity.py` |
| Methods Sec. 4.12 | pre-specified success criterion and clinical-relevance threshold | `evaluation/report.py` |
| Tables 1, 3, 4 and Figs. 1-4 | reported tables and figures | assembled in `evaluation/report.py`, `evaluation/silent_mode.py`, `evaluation/ascertainment.py` |

## Training

One command per arm; the experiment name selects the config file and the arm selects
the model variant.

```bash
python -m stagefm.cli.train experiment=main
python -m stagefm.cli.train experiment=main arm=finetuned_independent_heads
python -m stagefm.cli.train experiment=ablation_without_risk_control
python -m stagefm.cli.train experiment=baseline_unconstrained_posthoc
```

Under `torchrun`:

```bash
scripts/launch_train.sh main stagefm
```

Reported profile, from `configs/experiment/main.yaml`:

| Setting | Value |
| --- | --- |
| batch size | 8 examinations per device |
| gradient accumulation | 1 |
| world size | 4 accelerators on one node |
| effective batch | 32 examinations per optimiser step |
| epochs | at most 60, early stopping at patience 8 |
| optimiser | AdamW, decoupled weight decay 1e-4 |
| learning rate | 3e-4 for fusion and prediction parameters, 1e-4 for the adapters |
| schedule | cosine decay with a linear warmup over the first 500 steps |
| gradient clipping | 1.0 |
| precision | bf16 |
| encoder | frozen, low-rank rank 16 with alpha 32 on q, k, v, out |
| volume grid | 96 x 96 x 64 voxels at 1 mm isotropic spacing |
| modality dropout | 0.15 per non-imaging stream |

A run writes `run.json`, `thresholds.json`, `run_summary.txt` and atomic checkpoints
into `runs/<experiment>/<arm>/`. Thresholds are selected on the internal test split at
the end of training and frozen there; the external sites are read once, afterwards.

The unit-test smoke profile is `configs/experiment/_smoke.yaml`. It exists only so the
test suite and the verification pass can run a two-step training loop on a laptop, and
it is labelled as such in the file. Do not use it for reporting.

## Evaluation

```bash
python -m stagefm.cli.eval experiment=main arm=stagefm
python -m stagefm.cli.eval experiment=main arm=unconstrained_posthoc
scripts/launch_eval.sh supplementary_ascertainment_analysis
```

A run writes `evaluation.json` plus a plain-text `evaluation_summary.txt` carrying the
three axis areas under the curve, the exact-combination concordance, the weighted
kappa, the treatment-boundary discordance with its interval, the calibration error and
the feasibility rate, with the per-site breakdown underneath.

Expected values are not stated here. The cohort those numbers would come from is
private, so every cohort-level value is `NOT_RUN` until the pipeline is pointed at
real records, and the verification report says so explicitly rather than filling the
gap with a generated number. What can be stated is the pre-specified criterion the
manuscript fixed before the external evaluation, and which the code evaluates
directly: exact-combination concordance of at least 0.80 together with a four-class
tumour-axis area under the curve of at least 0.90, plus a treatment-boundary
discordance reduction of at least 3.0 absolute points against the control arm.

## Compute budget

The reported run used one node with four accelerators and 80 GB of memory per node,
totalling roughly 3,100 accelerator-hours across the five independent runs per
configuration plus the hyperparameter grid. None of these numbers is softened: a
single configuration at full training is a multi-day job on that hardware, and the
ablation, baseline and supplementary configurations multiply it.

Disk: the analysis set is small, but the auxiliary resources are not. The abdominal CT
vision-language dataset is several terabytes, the TotalSegmentator archive is a few
gigabytes, and the FLARE 2023 archives are over 300 GB. Budget accordingly, or run the
encoder fallback described below.

## Encoder availability

The released abdominal-CT vision-language encoder is distributed under a data-use
agreement and is not bundled. When `model.encoder.weights_path` is unset, the pipeline
uses the compact 3D vision transformer in `models/encoder.py`, which satisfies the
same token contract and carries the same attention-projection names the adapters bind
to. Which encoder was used is recorded in every run's metadata and in the verification
report, and it is listed in the deviations section of `claim_to_code.json`.

## Verification and status

The verification pass resolves the paper-claim mapping and executes the pipeline:

```bash
python -m stagefm.cli.verify
```

It writes `claim_to_code.json`, `verification_report.json`,
`verification_summary.txt` and `integrity_manifest.json`, and it re-derives the core
numerical procedures with independent code: explicit enumeration for the feasibility
projection, brute-force pairwise counting for the areas under the curve, a
hand-computed decomposition for the intraclass correlation, closed forms for the
decision metrics, and a permutation-and-bootstrap cross-check for the DeLong variance.
The verification code never calls the routine it is checking to produce its expected
value.

`verification_report.json` carries two statuses. `code_status` covers the provenance,
execution, algorithm, metric, statistic and environment families. `overall_status`
also includes the manuscript-arithmetic family, whose one failing entry records a
discrepancy inside the manuscript's own Table 4: applying the interaction-ratio
estimator to the four tabulated concordance rows gives 0.57, while the text reports
0.87. The discrepancy is reported rather than reconciled.

Deliberate departures from the manuscript, and the engineering defaults that stand in
where the manuscript is silent, are listed in the `deviations` array of
`claim_to_code.json`; each entry names its paper location and its justification.

## Licence and third-party material

The code is released under the Apache License 2.0; see `LICENSE`. Third-party
datasets, models and their licences are named in `THIRD_PARTY_NOTICES.txt`, and their
access conditions in `dataset_urls.txt`.
