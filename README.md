# artificial-data-gender

This repository contains the code to reproduce the experiments in the paper [Using Artificial French Data to Understand
the Emergence of Gender Bias in Transformer Language Models]().

Each experiment can be run on an individual Jupyter Notebook. 
All notebooks rely on the code in the folder `Lm4Ling` and on the file `full_vocab.json` containing vocabulary taken 
from the [Universal Dependencies](https://github.com/UniversalDependencies/UD_French-GSD) project to build the PCFGs.

`full_vocab.json` was built using the code in [generate_vocab.ipynb](https://github.com/lina-conti/artificial-data-gender/blob/main/generate_vocab.ipynb).

The implementation of a transformer language model is taken from [Benoît Crabbé](https://github.com/bencrabbe)'s `Lm4Ling` repository.

## Decoupling contextual gender information from static gender associations

The code to reproduce this experiment is in [exp_context_or_static.ipynb](https://github.com/lina-conti/artificial-data-gender/blob/main/exp_context_or_static.ipynb).

## Assignment of gender to epicene nouns

The code to reproduce this experiment is in [exp_context_or_static.ipynb](https://github.com/lina-conti/artificial-data-gender/blob/main/exp_context_or_static.ipynb).

## Impact of word frequency

The code to reproduce this experiment is also in [exp_epicenes.ipynb](https://github.com/lina-conti/artificial-data-gender/blob/main/exp_epicenes.ipynb).

## Influence of the gender distribution on default gender guessing 

The code to reproduce this experiment is in [exp_default_guess.ipynb](https://github.com/lina-conti/artificial-data-gender/blob/main/exp_default_guess.ipynb).
