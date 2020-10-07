# Generalization of Counterfactually-Augmented NLI Data

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

### Set-up

To set-up  an environment first install requirements with the following:

```
git clone https://github.com/wh629/CNLI-generalization.git
pip install -r jiant/requirements-dev.txt
```

Then install apex from:

`https://github.com/NVIDIA/apex`

### Description

#### General

You can use `get_all_exp.sh` in `run_scripts` to get Python commands for experiments using:

```
sh get_all_exp.sh roberta-base none
```

#### NYU Prince

Experiments are run on NYU's Prince HPC with Slurm. The following command can be used to generate commands to submit multiple jobs:

```
sh get_all_exp.sh roberta-base <absolute path to .sbatch file>
```

An example `.sbatch` is provided in `run_scripts` that requires updates to the `<env name>` and `<jiant path>`.



Commands for both set-ups can be found in the newly created `exp_scripts` directory as `submit_exp_<training data>-<validation data>_<time stamp>.sh`.

### License
Our code is released under the [MIT License](https://github.com/jiant-dev/jiant/blob/master/LICENSE).
