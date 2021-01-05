# BOCPDMS: Bayesian On-line Changepoint Detection with Model Selection

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/alan-turing-institute/bocpdms/master?filepath=examples%2FNile.ipynb)

This repository contains code from the _Bayesian On-line Changepoint Detection with Model Selection_ project.

## Table of contents

* [About BOCPDMS](#about-bocpdms)
* [Reproducible Research Champions](#reproducible-research-champions)
* [Installation instructions](#installation-instructions)
* [Running the examples](#running-the-examples)
* [Contributors](#contributors)


## About BOCPDMS

Bayesian On-line Changepoint Detection (BOCPD) is a discrete-time inference framework introduced in the statistics and machine learning community independently by [Fearnhead & Liu (2007)](https://doi.org/10.1111/j.1467-9868.2007.00601.x) and [Adams & MacKay (2007)](https://arxiv.org/abs/0710.3742). Taken together, both papers have generated in excess of 500 citations and inspired more research in this area. The method is popular because it is efficient and runs in constant time per observation processed. We are working on extending the inference paradigm in several ways:

- [x] Unifiying Fearnhead & Liu (2007) and Adams & MacKay (2007)¹
- [x] Multivariate analysis¹
- [x] Robust analysis²
- [ ] Continuous-time models
- [ ] Point processes

### Papers

¹Jeremias Knoblauch and Theodoros Damoulas. [Spatio-temporal Bayesian On-line Changepoint Detection](https://arxiv.org/abs/1805.05383), _International Conference on Machine Learning_ (2018).

²Jeremias Knoblauch, Jack Jewson and Theodoros Damoulas. [Doubly Robust Bayesian Inference for Non-Stationary Streaming Data with β-Divergences](https://arxiv.org/abs/1806.02261), arXiv:1806.02261 (2018).

### Code

The code in this repository was used in both papers, and we are currently working on splitting the two projects so that it is easier to reproduce the work in both the older¹ and newer² papers. You can track our progress on this in [issue \#14](https://github.com/alan-turing-institute/bocpdms/issues/14).

Until we close \#14, you may notice that the results from some of the examples are _robust_, but do not exactly _reproduce_ those from the earlier ICML paper. This is due to changes in the core classes, and in particular the hyperparameter optimisation process, between the publication of the two papers.

Want a preview of the ICML results? Take a look at the updated demo in the branch associated with issue \#14 on Binder: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/alan-turing-institute/bocpdms/feature/14-remove-nips?filepath=examples%2FNile.ipynb)

## Reproducible Research Champions

In May 2018, Theo Damoulas was selected as one of the Alan Turing Institute's Reproducible Research Champions - academics who encourage and promote reproducible research through their own work, and who want to take their latest project to the "next level" of reproducibility.

The Reproducible Research programme at the Turing is led by Kirstie Whitaker and Martin O'Reilly, with the Champions project also involving members of the Research Engineering Group.

Each of the Champions' projects will receive several weeks of support from the Research Engineering Group throughout Summer 2018; during this time, we will work on the project together with Jeremias and Theo and will track our efforts in this repository. Given our focus on reproducibility, we obviously won't be changing any of the code's functionality - but we will make it easier for you to install, use and test out your own ideas with the BOCPDMS methodology.

You can keep track of our progress through the Issues tab, and find out more about the Turing's Reproducible Research Champions project [here](https://github.com/alan-turing-institute/ReproducibleResearchResources).

## Installation instructions

1. Clone this repository (see [this useful guide](https://help.github.com/articles/cloning-a-repository/) to get started)
2. Change to the repository directory on your local machine
3. \[Optional] Create a new virtual environment for this project (see [*why use a virtual environment*](#why-use-a-virtual-environment) below)
4. Install the required packages using `pip install -r requirements.txt`
5. \[Optional] Verify that everything is working by running the tests (see [*run the tests*](#run-the-tests) below)


### Why use a virtual environment?

A virtual environment is an isolated instance of python that has its own separately managed set of installed libraries (dependencies).
Creating a separate virtual environment for each project you are reproducing has the following advantages:

  1. It ensures you are using **only** the libraries specified by the authors.
    This verifies that they have provided **all** the information about the required libraries necessary to reproduce their work and that you are not accidentally relying on previously installed versions of common libraries.
  2. It ensures that you are using the **same versions** of the libraries specified by the authors.
     This ensures that a failure to reproduce is not caused by changes to libraries made between the authors publishing their project and you attempting to reproduce it.
  3. It ensures that none of the libraries required for the project interfere with the libraries installed in the standard python environment you use for your day to day work.

You can create a new virtual environment using python's built-in `venv` command (see [*instructions with venv*](#instructions-with-virtualenv) below), or with `conda` ([*instructions with conda*](#instructions-with-conda)).

Note that this project will not run a virtual environment created using `virtualenv`.
This is due to a [known issue with matplotlib and virtualenv](https://matplotlib.org/faq/osx_framework.html).


#### Instructions with conda

For more detailed instructions, check out the conda [managing environments](https://conda.io/docs/user-guide/tasks/manage-environments.html) documentation.
Hopefully though, the following commands are enough to get you started.

From inside the `bocpdms` folder on your computer:

```
conda create -n bocpdms python=3.7
conda activate bocpdms
pip install -r requirements.txt
```

If you want to use `jupyter lab` with this new environment, you should also run the following command so you can see this new `bocpdms` kernel :sparkles:
```
conda install -c conda-forge jupyterlab
conda install nb_conda_kernels
```
You can then launch Jupyter Lab using `jupyter lab` while your virtual environment is active.


#### Instructions with venv

For OSX or Linux, you can use `venv` instead of `conda`.
For more detailed instructions, check out the [venv documentation](https://docs.python.org/3/library/venv.html) documentation.
Hopefully though, the following commands are enough to get you started.

From inside the `bocpdms` folder on your computer:

Feel free to change the folder the virtual environemnt is created in by replacing `~/.virtualenvs/bocpdms` with a path of your choice in both commands.
```
python3 -m venv ~/.virtualenvs/bocpdms
source ~/.virtualenvs/bocpdms/bin/activate
pip install -r requirements.txt
```
If you want to use jupyter lab with this new environment, you should also run the following command so you can see this new `bocpdms` kernel :sparkles:
```
pip install jupyterlab
pip install ipykernel
ipython kernel install --user --name=venv-bocpdms
```
You can then launch Jupyter Lab using `jupyter lab` while your virtual environment is active.


### Run the tests

From the repository directory run `python -m pytest`.

This will run all the tests in the `tests` folder of the project.

You should see the following celebratory message :tada::sparkles::cake:

```
============================= test session starts =============================
platform win32 -- Python 3.7.0, pytest-3.7.1, py-1.7.0, pluggy-0.8.0
rootdir: \path\to\your\version\of\bocpdms, inifile:
collected 6 items

tests\test_Evaluation_tool.py .....                                      [ 83%]
tests\test_nile_example.py .                                             [100%]

========================== 6 passed in 17.83 seconds ==========================
```


## Running the examples

You can jump directly to an interactive demo of the Nile example by clicking on this Binder button:
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/alan-turing-institute/bocpdms/master?filepath=examples%2FNile.ipynb)

To run from the command line, first activate your virtual environment as described above. You can then run, for example,
```
python nile_ICML18.py
```
and
```
python paper_pictures_nileData.py
```
to generate the figure(s). Recently, we have started to add further options that let you change various parameters from the command line. These are currently available for the Nile river height and bee waggle dance examples (although you can find this functionality for some of the other scripts in their respective [branches](https://github.com/alan-turing-institute/bocpdms/branches)). You can see the various options with the following commands:
```
python nile_ICML18.py --help
python bee_waggle_ICML18.py --help
```

## Contributors

Thank you to the following for their contributions to this project:
- Jeremias Knoblauch
- Theo Damoulas
- Kirstie Whitaker
- Martin O'Reilly
- Louise Bowler
