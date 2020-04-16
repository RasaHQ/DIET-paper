# DIET-paper

Source code to reproduce results of our paper "DIET: Lightweight Language Understanding for Dialogue Systems"

Steps to follow:

(1) We used Rasa for running the experiments. 
You first need to clone the repository and checkout the branch `diet-paper`.

```bash
git clone https://github.com/RasaHQ/rasa
git checkout diet-paper
```

(2) Execute the `run.sh` script to train and test models using the best model configuration on the ATIS, SNIPS, and
NLU Evaluation Dataset.

```bash
sh run.sh
```

(3) The experiments results can be found in the folder `experiments`.