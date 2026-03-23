## Overview

This repository contains the implementation of our proposed distillation method.
The codebase is **built on top of two existing frameworks**:

- **DistillM**
- **DistillM2**

Our implementation extends these frameworks. Data preprocessing steps and environment configurations are inherited from the original projects.

**Important**: Detailed installation instructions are provided in the original repositories and are not duplicated here.

---

## Dependencies and Environment Setup

This artifact reuses the environments of the base projects (DistillM / DistillM2).

Please follow the official setup instructions in:

- `distillm-fdd/README.md`
- `distillm-2-master/README.md`

In addition, this project requires **spaCy** and the English language model
`en_core_web_sm` for text processing.

Install the required dependencies as follows:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

Once either DistillM or DistillM2 is successfully installed, this artifact can be executed directly.

---

## Running

We provide example scripts to reproduce the main experiments with GPT-2 120M reported in the paper.

All experiments are executed **with datasets that have already been preprocessed and prepared**
following the instructions of the base frameworks (DistillM / DistillM2).

All experiments reported in the paper were conducted on **a single NVIDIA A100 GPU with 40GB memory**.

Users should ensure that the dataset paths specified in the scripts point to the prepared data directories produced by the original repositories.

### DistillM-based + FDD-based Experiment

```bash
bash run_distillm_fdd_example.sh
```
### DistillM2-based Experiment
```bash
bash run_distillm2_example.sh
```
