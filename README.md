# LISA: Linguistically-Informed Self-Attention

![](./lisa.jpg)

Requirements:
----
- Python 3.
- TensorFlow 1.10+


Data setup:
----
1. Get pre-trained word embeddings (GloVe):
    ```
    wget -P embeddings http://nlp.stanford.edu/data/glove.6B.zip
    unzip -j embeddings/glove.6B.zip glove.6B.100d.txt -d embeddings
    ```
2. Get CoNLL-2005 data:


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