# Change Point Detection via Multivariate Singular Spectrum Analysis

This is the code submission associated with the paper "Change Point Detection via Multivariate Singular Spectrum Analysis"


## Requirements

* python 3.7
* python packages in `requirements.txt`
* R [for binseg algorithm only]


## Hardware Requirements

* GPU device [for klcpd algorithm only]


## Datasets

We provide the synthetic and benchmark datasets used in the experiments. The datasets can be be found in the `data` folder.


## Running CPD 

To run change point detection experiments use the notebook `run_experiment.ipynb`.

Make the following selections: 	
* `algorithm` from {"binseg", "microsoft_ssa", "klcpd", "bocpdms", "mssa", "mssa_mw"}
* `dataset` from {"energy", "mean", "mixed", "frequency", "beedance", "hasc", "occupancy", "yahoo"}
* `time_series` from {"all", <time_series> [check `config.py` for naming]}
* `params_type` from {"default", "best", "custom" [define params dict for this option]}


