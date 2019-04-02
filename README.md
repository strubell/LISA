# LISA: Linguistically-Informed Self-Attention

![](./lisa.jpg)

This is a work-in-progress, but much-improved, re-implementation of the 
linguistically-informed self-attention (LISA) model described in the following paper:
> Emma Strubell, Patrick Verga, Daniel Andor, David Weiss, and Andrew McCallum. [Linguistically-Informed 
> Self-Attention for Semantic Role Labeling](https://arxiv.org/abs/1804.08199). 
> *Conference on Empirical Methods in Natural Language Processing (EMNLP)*. 
> Brussels, Belgium. October 2018. 

**To exactly replicate the results in the paper at the cost of an unpleasantly hacky codebase, you
can use the [original LISA code here](https://github.com/strubell/LISA-v1).**

Requirements:
----
- \>= Python 3.6
- \>= TensorFlow 1.9 (tested up to 1.12)

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

Evaluate an exported model:
----
To evaluate the best<sup id="f1">[1](#f1)</sup> checkpoint so far, saved in the directory `model` (with id 1554216594):
```
bin/evaluate-exported.sh config/conll05-lisa.conf --save_dir model/export/best_exporter/1554216594
```

# Custom configuration [WIP]

LISA model configuration is defined through a combination of configuration files. A top-level config defines a specific model configuration and dataset by setting other configurations. Top-level configs are written in bash, and bottom-level configs are written in json. Here is an example top-level config, [`conll05-lisa.conf`](config/conll05-lisa.conf), which defines the basic LISA model and CoNLL-2005 data:
```
# use CoNLL-2005 data  
source config/conll05.conf  
  
# take glove embeddings as input  
model_configs=config/model_configs/glove_basic.json  
  
# joint pos/predicate layer, parse heads and labels, and srl  
task_configs="config/task_configs/joint_pos_predicate.json,config/task_configs/parse_heads.json,config/task_configs/parse_labels.json,config/task_configs/srl.json"  
  
# use parse in attention  
attention_configs="config/attention_configs/parse_attention.json"  
  
# specify the layers  
layer_configs="config/layer_configs/lisa_layers.json"
```
And the top-level data config for the CoNLL-2005 dataset that it loads, [`conll05.conf`](config/conll05.conf):
```
data_config=config/data_configs/conll05.json  
data_dir=$DATA_DIR/conll05st-release-new  
train_files=$data_dir/train-set.gz.parse.sdeps.combined.bio  
dev_files=$data_dir/dev-set.gz.parse.sdeps.combined.bio  
test_files=$data_dir/test.wsj.gz.parse.sdeps.combined.bio,$data_dir/test.brown.gz.parse.sdeps.combined.bio
```
Note that `$DATA_DIR` is a bash global variable, but all the other variables are defined in these configs.

There are five types of bottom-level configurations, specifying different aspects of the model:
- [**data configs**](#data-configs): Data configs define a mapping from columns in a one-word-per-line formatted file (e.g. the CoNLL-X format) to named features and labels that will be provided to the model as batches.
- [**model configs**](#model-configs): Model configs define hyperparameters, both *model hyperparameters*, like various embedding dimensions, and *optimization hyperparameters*, like learning rate. Optimization hyperparameters can be reset at the command line using the `hparams` command line parameter, which takes a comma-separated list of `name=value` hyperparameter settings. Model hyperparameters cannot be redefined in this way, since this would invalidate a serialized model.
- [**task configs**](#task-configs): Task configs define a task: label, evaluation, and how predictions are formed from the model. Each task (e.g. SRL, parse edges, parse labels) should have its own task config.
- [**layer configs**](#layer-configs):  Layer configs attach tasks to layers, defining which layer representations should be trained to predict named labels (from the data config). The number of layers in the model is determined by the maximum depth listed in layer configs.
- [**attention configs**](#attention-configs) (optional): Attention configs define special attention functions which replace attention heads, i.e. syntactically-informed self attention. Omitting any attention configs results in a model performing simple single- or multi-task learning.

How these different configuration files work is specified in more detail below.

## Data configs

An full example data config can be seen here: [`conll05.json`](config/data_configs/conll05.json). 

Each top-level entry in the json defines a named feature or label that will be provided to the model. The following table describes the possible parameters for configuring how each input is interpreted.

|     Field      |Type          |Description       | Default value |       
|----------------|----------|------------------------|---|
| `conll_idx` | int or list | Column in the data file corresponding to this input.  | N/A (required) |
| `vocab`     | string | Name of the vocabulary used to map this (string) input to int.| None (output of converter is int) |
| `type`      | string          | Type of `conll_idx`. Possible types are: range, other (int/list). "range" can be used to specify that a variable-length range of columns should be read in at once and passed to the converter. Otherwise, the given single int or list of columns is read in and passed to the converter. | "other" (int/list)|
| `feature`	  | boolean | Whether this input should be used as a feature, i.e. provided to the model as input. | false |
| `label`     | boolean| Whether this input should be used as a label, i.e. provided to the model as a label. | false |
| `updatable` | boolean | Whether this vocab should be updated after its initial creation (i.e. after creating a vocab based on the training data). | false |
| `converter` | json| A json object defining a function (name and, optionally, parameters) for converting the raw input. These functions are defined in [`src/data_converters.py`](src/data_converters.py). | `idx_list_converter` |
| `oov` | boolean | Whether an `OOV` entry should be added to this input's vocabulary. | false |

### Converters
The data config specifies a converter function and vocabulary for each desired column in the input data file. For each entry in the data config and each line in the input file, the column values specified by `conll_idx` are read in and provided to the given converter. Data generators, which take the data config and data file as input to perform this mapping, are defined in [`src/data_generator.py`](src/data_generator.py). 

New converter functions can be defined in [`src/data_converters.py`](src/data_converters.py). At a minimum, every converter function takes two parameters: `split_line`, the current line in the data file split by whitespace, and `idx`, the value of `conll_idx`. Converters may also take additional parameters, whose values are defined via the `params` field in the converter json object. The output of a converter is a list of strings.

For example, the default converter, `idx_list_converter`, simply takes a single column index or list of indices and returns a list containing the corresponding column values in the input file:
```python
def idx_list_converter(split_line, idx):
  if isinstance(idx, int):
    return [split_line[idx]]
  return [split_line[i] for i in idx]
```

### Vocabs

When a vocab is specified for an entry in the data config, that vocab is used to map the string output of the converter to integer values suitable for features/labels in a TensorFlow model.<sup id="f2">[2](#f2)</sup> This mapping occurs in the `map_strings_to_ints` function in [`src/dataset.py`](src/dataset.py). 

- TODO: vocab initialization
- TODO: pre-trained word embeddings

## Model configs
TODO

## Layer configs
TODO

## Task configs
TODO

## Attention configs
TODO

# Footnotes
<b id="f1">1</b>: "Best" is determined by `best_eval_key`, with default value for a given dataset in the top-level data config, e.g. [`config/conll05.conf`](config/conll05.conf). The value of `best_eval_key` must correspond to a named eval under the `eval_fns` entry in a [task config](#task-configs). [↩︎](#f1)

<b id="f2">2</b>: If no vocab is specified, then it's assumed that the output of the converter can be interpreted as an integer. [↩︎](#f2)
