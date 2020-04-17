# DIET-paper

Source code to reproduce results of our paper "DIET: Lightweight Language Understanding for Dialogue Systems"

In order to reproduce the experiments results, execute the following steps:

(1) We used Rasa for running the experiments. 
You first need to clone the repository, checkout the branch `diet-paper` and install Rasa.

```bash
git clone https://github.com/RasaHQ/rasa
git checkout diet-paper
make install-full
```

(2) Execute the `run.sh` script to train and test models using the best model configuration on ATIS, SNIPS, and
NLU Evaluation Dataset. Make sure you are at the root directory of this repository before executing the script.

```bash
sh run.sh
```

Consider executing the experiments on a machine with a GPU to speedup the experiments.

(3) The experiments results can be found in the folder `experiments`.