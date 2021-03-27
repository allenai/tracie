# Introduction
This is the data and code repository for our NAACL 2021 paper "Temporal Reasoning on Implicit Events from Distant Supervision". 
We include the dataset TRACIE and the proposed reasoning model SymTime here.

# TRACIE
TRACIE is our crowdsourced dataset that is designed to evaluate system's ability for temporal reasoning over implicit events. 
## Dataset
We include the dataset under `data/`. We provide two splits: IID (`data/iid/`) and Uniform-Prior (`data/uniform-prior`).
The data is in a NLI format with each line being `event: [query] story: [context] \t answer: [label] \n`.
## Leaderboard
Coming soon.

# Models and Experiments
We provide our codebase for our proposed model, PtnTime and SymTime with code to reproduce the experiments reported in the paper. 
Specific numbers may fluctuate depending on different GPU and driver versions, but relative relationships should hold.
The original experiments were run on RTX 8000 GPUs and CUDA10.

## PtnTime
PtnTime is simply the T5 sequence-to-sequence implementation from Huggingface's transformer (v2.11.0).
Instead of Google's pre-trained weights, PtnTime uses different model weights for T5, which is the only difference comparing to the T5-large baseline.

### Pre-trained Model
Download the entire directory `ptntime-pretrained-model` from TBA
and put it under `code/models/ptntime/` 

### Run experiments
We provide the source code as both a Dockerfile and shell scripts. We introduce how to run the shell scripts here.

- Work under the directory `code/models/ptntime` (This is very important as we refer all paths relative to this working directory below.)
- Install requirements by `pip install -r requirements.txt`

To run the T5-large baseline, use `sh run_t5_large_baseline_on_uniform_prior.sh`

To run the PtnTime pre-trained model, use `sh run_ptntime_on_uniform_prior.sh`

Random seeds and data sources can be replaced within the provided shell scripts.

Both scripts will create a result file `experiment_result/eval_results_lm.txt`. To evaluate, run `python evaluator.py`

## SymTime
The SymTime model uses two separate T5 models and a custom inference.

### Pre-trained Model
Download the entire directory `symtime-pretrained-model` from TBA and put it under `code/models/symtime`

### Run experiments
- Word under the directory `code/models/symtim`
- Install requirements by `pip install -r requirements.txt`
- `sh run_symtim_on_uniform_prior.sh` runs the SymTime model on the uniform prior split
- `python evaluator.py` prints the results

## Pretraining
TBD

# Citation
See the following paper: 
```
@inproceedings{ZRNKSR21,
    author = {Ben Zhou, Kyle Richardson, Qiang Ning, Tushar Khot, Ashish Sabharwal and Dan Roth},
    title = {Temporal Reasoning on Implicit Events from Distant Supervision},
    booktitle = {NAACL},
    year = {2021},
}
```