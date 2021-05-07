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
The numbers reported in the paper were run on RTX 8000 and CUDA10, with reproducible Docker images hosted on [Beaker](https://beaker.org/).

We observed slight fluctuations of numbers on different GPUs and driver/CUDA versions during the camera-ready stage, but the relative relationships hold.

## PtnTime
PtnTime is simply the T5 sequence-to-sequence implementation from Huggingface's transformer (v2.11.0).
Instead of Google's pre-trained weights, PtnTime uses different model weights for T5, which is the only difference comparing to the T5-large baseline.

### Pre-trained Model
Download the entire directory `ptntime-pretrained-model` from [Google Drive](https://drive.google.com/drive/folders/1GirBYMWHJ13zqKl5qPcTjJQNJVtCfVaP?usp=sharing)
and put it under `code/models/ptntime/` 

### Run experiments
We provide the source code as both a Dockerfile and shell scripts. We introduce how to run the shell scripts here.

- Work under the directory `code/models/ptntime` (This is very important as we refer all paths relative to this working directory below.)
- Install requirements by `pip install -r requirements.txt`

To run the T5-large baseline, use `sh run_t5_large_baseline_on_uniform_prior.sh`

To run the PtnTime pre-trained model, use `sh run_ptntime_on_uniform_prior.sh`

Random seeds and data sources can be replaced within the provided shell scripts.

Both scripts will create a result file `experiment_result/eval_results_lm.txt`. To evaluate, run `python evaluator.py`.
We provide the predictions from our experiments with 3 seeds (`10/20/30.txt`) and uniform-prior settings (`up_{seed}.txt`) under `experiment_result`.

The MATRES experiment can also be ran with this PtnTime with data under `data/matres`.

## SymTime
The SymTime model uses two separate T5 models and a custom inference.

### Pre-trained Model
Download the entire directory `symtime-pretrained-model` from [Google Drive](https://drive.google.com/drive/folders/1GirBYMWHJ13zqKl5qPcTjJQNJVtCfVaP?usp=sharing) and put it under `code/models/symtime`

### Run experiments
- Word under the directory `code/models/symtim`
- Install requirements by `pip install -r requirements.txt`
- `sh run_symtim_on_uniform_prior.sh` runs the SymTime model on the uniform prior split
- `python evaluator.py` prints the results

Note: the input data format is different from that of PtnTime, which is provided under `data/iid-symbolic-format` and `data/uniform-prior-symbolic-format`.
We similarly provide our model outputs under `experiment_results`.

## Pretraining
Two separately pre-trained models were used in the paper. They were trained with the PtnTime model (basic T5 seq2seq).
We provide the data for pre-training in the `pretrain-data` directory ([Google Drive](https://drive.google.com/drive/folders/1GirBYMWHJ13zqKl5qPcTjJQNJVtCfVaP?usp=sharing)): 
- `pretrain-distance.txt`: Start times and (in some instances) distances between start times. One instance per line.
- `pretrain-duration.txt`: Durations. One instance per line.

Please refer to the paper for formats.

## Extraction
The extraction for the start time comparisons relies on AllenNLP's SRL model. We provide the parsing scripts under `code/extractions`.
- `code/extractions/extractor_before_after.py` parses SRL results to get the within-sentence extractions
- `code/extractions/extractor_distance.py` parses SRL results over paragraphs to get the cross-sentence extractions.

Both scripts require additional SRL results, which we can provide on demand due to size limits.


# Citation
See the following paper: 
```
@inproceedings{ZRNKSR21,
    author = {Ben Zhou and Kyle Richardson and Qiang Ning and Tushar Khot and Ashish Sabharwal and Dan Roth},
    title = {Temporal Reasoning on Implicit Events from Distant Supervision},
    booktitle = {NAACL},
    year = {2021},
}
```
