# LISA: Linguistically-Informed Self-Attention

![](./lisa.jpg)

This is a work-in-progress, but much-improved, re-implementation of the 
linguistically-informed self-attention (LISA) model described in the following paper:
> Emma Strubell, Patrick Verga, Daniel Andor, David Weiss, and Andrew McCallum. [Linguistically-Informed 
> Self-Attention for Semantic Role Labeling](https://arxiv.org/abs/1804.08199). 
> *Conference on Empirical Methods in Natural Language Processing (EMNLP)*. 
> Brussels, Belgium. October 2018. 

To exactly replicate the results in the paper at the cost of an unpleasantly hacky codebase, you
can use the [original LISA code here](https://github.com/strubell/LISA-v1).

Requirements:
----
- \>= Python 3.6
- \>= TensorFlow 1.10

Quick start:
============

Data setup (CoNLL-2005):
----
1. Get pre-trained word embeddings (GloVe):
    ```
    wget -P embeddings http://nlp.stanford.edu/data/glove.6B.zip
    unzip -j embeddings/glove.6B.zip glove.6B.100d.txt -d embeddings
    ```
2. Get CoNLL-2005 data in the right format using [this repo](https://github.com/strubell/preprocess-conll05). 
Follow the instructions all the way through [further preprocessing](https://github.com/strubell/preprocess-conll05#further-pre-processing-eg-for-lisa).
3. Make sure the correct data paths are set in `config/conll05.conf`

Train a model:
----
To train a model with save directory `model` using the configuration `conll05-lisa.conf`:
```
bin/train.sh config/conll05-lisa.conf --save_dir model
```

Evaluate a model:
----
To evaluate the latest checkpoint saved in the directory `model`:
```
bin/evaluate.sh config/conll05-lisa.conf --save_dir model
```