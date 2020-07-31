# Multi-Directional Rule Set Learning

This project contains a Python module for learning **multi-directional rule sets**, accompanying our paper:
> Schouterden J., Davis J., Blockeel H.: *Multi-Directional Rule Set Learning.* To be presented at: Discovery Science 2020

__________________________________
[Abstract](https://github.com/joschout/Multi-Directional_Rule_Set_Learning#abstract) -
[Basic use](https://github.com/joschout/Multi-Directional_Rule_Set_Learning#basic-use) -
[Experiments](https://github.com/joschout/Multi-Directional_Rule_Set_Learning#experiments) -
[Dependencies](https://github.com/joschout/Multi-Directional_Rule_Set_Learning#dependencies) -
[References](https://github.com/joschout/Multi-Directional_Rule_Set_Learning#references) 
_________________


## Abstract
The following is the abstract of our paper:

>A rule set is a type of classifier that, given attributes X, predicts a target Y. Its main advantage over other types of classifiers is its simplicity and interpretability. A practical challenge is that the end user of a rule set does not always know in advance which target will need to be predicted. One way to deal with this is to learn a multi-directional rule set, which can predict any attribute from all others. 
An individual rule in such a multi-directional rule set can have multiple targets in its head, and thus be used to predict any one of these.
>
>Compared to the naive approach of learning one rule set for each possible target and merging them, a multi-directional rule set containing multi-target rules is potentially smaller and more interpretable. Training a multi-directional rule set involves two key steps: generating candidate rules and selecting rules. However, the best way to tackle these steps remains an open question.
>
>In this paper, we investigate the effect of using Random Forests as candidate rule generators and 
propose two new approaches for selecting rules with multi-target heads:
MIDS, a generalization of the recent single-target IDS approach, and RR, a new simple algorithm focusing only on predictive performance. 
>
>Our experiments indicate that (1) using multi-target rules leads to smaller rule sets with a similar predictive performance, (2) using Forest-derived rules instead of association rules leads to rule sets of similar quality, and (3) RR outperforms MIDS, underlining the usefulness of simple selection objectives.

## Basic use

The basic use of IDS, RR and MIDS is illustrated by the following Jupyter notebooks:

* [Single-target Interpretable Decision Sets (IDS)](./notebooks/basic_use/ids_on_titanic.ipynb)
* [Round Robin (RR)](./notebooks/basic_use/rr_on_titanic.ipynb)
* [Multi-directional IDS (MIDS)](./notebooks/basic_use/mids_on_titanic.ipynb)

These notebooks illustrate these rule set classifiers on [the Titanic toy dataset that was provided with the reference IDS implementation](https://github.com/lvhimabindu) accompanying the original IDS paper.


To use this project as a Python module, you can install it in your Python environment after cloning this repository as follows:
  ```shell
  git clone https://github.com/joschout/Multi-Directional_Rule_Set_Learning.git
  cd Multi-Directional_Rule_Set_Learning/
  python setup.py install develop --user
  ```

## Experiments

In our paper, we include two sets of experiments. Here, we describe how to reproduce these experiments. We use data generated using the scripts from the [arcBench benchmarking suit](https://github.com/kliegr/arcBench), by Tomas Kliegr. You can find the data we used [in this repository, in the `data` directory](./data).

### 1. Comparing models generated from association rules and Random Forest derived rules.

In our first experiment, we compared two different single-target candidate rule sets our of which an asssociative classifier can be selected:
1. single-target association rules, and
2. single-target rules derived from Random Forest trees.

For this experiment, single-target IDS is used as the rule selection algorithm. (Note: in our experiments, we use our MIDS implementation, which corresponds to IDS when given single-target rules.)
The code for this experiment can be found in [`experiments/e1_st_association_vs_tree_rules`](./experiments/e1_st_association_vs_tree_rules).
 
To reproduce this experiment, you can do the following for each candidate rule set type:
* When considering single-target **association rules** as the candidate rule set:
    1. [Mine single-target association rules.](./experiments/e1_st_association_vs_tree_rules/rule_mining/single_target_car_mining_ifo_confidence_level.py)
    2. [Fit an AR-IDS model.](./experiments/e1_st_association_vs_tree_rules/model_induction/single_target_car_mids_model_induction.py) That is, use IDS to select a subset of the candidate single-target association rules.
    3. [Evaluate the AR-IDS model on the test data](./experiments/e1_st_association_vs_tree_rules/model_evaluation/single_target_car_mids_model_evaluation.py), measuring both predictive performance and interpretability.
* When considering single-target **rules derived from random forests (i.e. decision trees)** as the candidate rule set:
    1. [Generate rules from single-target Random Forests.](./experiments/e1_st_association_vs_tree_rules/rule_mining/single_target_tree_rule_generation_ifo_confidence_bound.py)
    2. [Fit a T-IDS model.](./experiments/e1_st_association_vs_tree_rules/model_induction/single_target_tree_mids_model_induction.py) That is, use IDS to select a subset of the candidate rules derived from single-target Random Forest trees.
    3. [Evaluate the T-IDS model on the test data](./experiments/e1_st_association_vs_tree_rules/model_evaluation/single_target_tree_mids_model_evaluation.py), measuring both predictive performance and interpretability.

### 2. Comparing multi-directional model generated from multi-target and single-target tree rules

In our second experiment, we compare multi-directional models selected from the following candidate rule sets:
1. multi-target rules derived from Random Forest trees, and
2. single-target rules defived from Random Forest trees.

From the multi-target rules, we fit two multi-directional models:
 * a Round Robin model, and
 * a MIDS model.
 
 From the single-target rules, we fit an ensemble of single-target IDS models.
The code for this experiment can be found in [`experiments/e2_multi_directional_model_comparison`](./experiments/e2_multi_directional_model_comparison).

To reproduce this experiments, you can do the following steps for each rule type:

* When using multi-target rules:
    1. [Generate multi-target rules from multi-target Random Forest trees.](./experiments/e2_multi_directional_model_comparison/rule_mining/mine_multi_target_rules_from_random_forests2.py)
    2. Choose which a rule selector able to select multi-directonal rules:
        * When using **Round Robin** as the rule selector, do:
            1. [Fit a multi-directional RR model.](./experiments/e2_multi_directional_model_comparison/model_induction/round_robin_tree_based_model_induction.py)
            2. [Evaluate the RR model on the test data.](./experiments/e2_multi_directional_model_comparison/model_evaluation/round_robin_tree_based_model_evaluation.py)
        * When using **MIDS** as the rule selector:
            1. [Fit a multi-directional MIDS model.](./experiments/e2_multi_directional_model_comparison/model_induction/mids_tree_based_model_induction.py)
            2. [Evaluate the MIDS model on the test data.](./experiments/e2_multi_directional_model_comparison/model_evaluation/mids_tree_based_model_evaluation.py)

* When using single-target rules:
    1. [Genearte single-target rules derived from single-target Random Forest trees, for each attribute in the dataset.](./experiments/e2_multi_directional_model_comparison/rule_mining/single_target_tree_based_rule_generation.py) This results in one candidate rule set per attribute.
    2. [Fit a single-target IDS model for each attribute in the dataset](./experiments/e2_multi_directional_model_comparison/model_induction/single_target_tree_mids_model_induction.py) This results in one single-target IDS model per attribute.
    3. [Merge the single-target IDS models into one ensemble (eIDS) model, and evaluate it on the test data.](./experiments/e2_multi_directional_model_comparison/eids_model_merging/single_target_tree_mids_model_merging.py)


## Dependencies

Depending on what you will use, you need to install some of the following packages. Note: we assume you have a recent Python 3 distribution installed (we used Python 3.8). Our installation instructions assume the use of a Unix shell.

* [*submodmax*](https://github.com/joschout/SubmodularMaximization), for unconstrained submodular maximization of the (M)IDS objective functions. This package is required for our versions of single-target IDS and Multi-directional IDS, as it contains the algorithms used for finding a locally optimal rule set. You can install it as follows:
  ```shell
  git clone https://github.com/joschout/SubmodularMaximization.git
  cd SubmodularMaximization/
  python setup.py install develop --user
  ```
* [PyFIM](https://borgelt.net/pyfim.html), by Christian Borgelt. This package is used for frequent itemset mining and (single-target) association rule mining, and is a dependency for pyARC. We downloaded the precompiled version and added it to our conda environment. This package is necessary wherever `import fim` is used.
* [pyARC](https://github.com/jirifilip/pyARC), by Jiří Filip. This package provides a Python implementation of the *Classification Based on Association Rules (CBA)* algorithn, which is one of the oldest *associative classifiers*. This package is a requirement for pyIDS. We make use of some of its data structures, and base some code snippets on theirs. *Note: there seems to be an error in the pyARC pip package, making the `QuantitativeDataFrame` class unavailable. Thus, we recommend installing it directly from the repository.*
  ```shell
  git clone https://github.com/jirifilip/pyARC.git
  cd pyARC/
  python setup.py install develop --user
  ```
* [pyIDS](https://github.com/jirifilip/pyIDS), by Jiří Filip and Tomas Kliegr. This package provides a great reimplementation of *Interpretable Decision Sets (IDS)*. We include a reworked IDS implementation in this repository, based on and using classes from pyIDS. To install pyIDS, run:
  ```shell
  git clone https://github.com/jirifilip/pyIDS.git
  cd pyIDS  
  ```
  Next, copy our the `install_utls/pyIDS/setup.py` to the `pyIDS` directory and run:
  ```shell
   python setup.py install develop --user
   ```
* *MLxtend* is a Python library with an implementation of *FP-growth* that allows to extract  *single-target class association rules*. Most association rule mining implementations only allow to mine single-target rules for a given target attribute, out of efficiency considerations. We forked MLxtend and modified it to also generate *multi-target* association rules. [Our fork can be found here](https://github.com/joschout/mlxtend/), while [the regular source code can be found here](https://github.com/rasbt/mlxtend). To install our fork, run:
  ```shell
  git clone https://github.com/joschout/mlxtend.git
  cd mlxtend/
  python setup.py install develop --user
  ```

* *gzip* and *jsonpickle* ([code](https://github.com/jsonpickle/jsonpickle), [docs](https://jsonpickle.readthedocs.io/en/latest/)) are used to save learned rule sets to disk.
* [tabulate](https://github.com/astanin/python-tabulate) is used to pretty-print tabular data, such as the different subfunction values of the (M)IDS objective function.
* [Apyori](https://github.com/ymoch/apyori), by Yu Mochizuki. Apriori implementation completely in Python.
* STAC: Statistical Tests for Algorithms Comparisons. ([Website](https://tec.citius.usc.es/stac/), [code](https://gitlab.citius.usc.es/ismael.rodriguez/stac/), [doc](https://tec.citius.usc.es/stac/doc/index.html), [paper PDF](http://persoal.citius.usc.es/manuel.mucientes/pubs/Rodriguez-Fdez15_fuzz-ieee-stac.pdf))
* graphviz, for visualizing decision trees during decision-tree-to-rule conversion.
* [bidict](https://github.com/jab/bidict), used to encode the training data during association rule minin. This way, large strings don't have to be used as data. `pip install bidict`

## References

This repository accompanies our paper:

> Schouterden J., Davis J., Blockeel H.: *Multi-Directional Rule Set Learning.* To be presented at: Discovery Science 2020


The original Interpretable Decision Sets algorithm was proposed in:
> Lakkaraju, H., Bach, S. H., & Leskovec, J. (2016). Interpretable Decision Sets: A Joint Framework for Description and Prediction. In Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1675–1684). New York, NY, USA: ACM. https://doi.org/10.1145/2939672.2939874

[The reference IDS implementation associated with the original IDS paper can be found here.](https://github.com/lvhimabindu) While it includes code to learn an IDS model, there is no code to actually apply the model, and no code to replicate the experiments from the paper. A great re-implementation of IDS by Jiri Filip and Tomas Kliegr called [PyIDS can be found here](https://github.com/jirifilip/pyIDS), and is described here: 
> Jiri Filip, Tomas Kliegr. PyIDS - Python Implementation of Interpretable Decision Sets Algorithm by Lakkaraju et al, 2016. RuleML+RR2019@Rule Challenge 2019. http://ceur-ws.org/Vol-2438/paper8.pdf

Our experiments use data from the UCI machine learning repository, modified for association rule learning using the arcBench benchmarking suite, which was proposed by Tomas Kriegr in:
> Kliegr, Tomas. Quantitative CBA: Small and Comprehensible Association Rule Classification Models. arXiv preprint arXiv:1711.10166, 2017.

For comparing our results, we use STAC, a great tool for statistically comparing the performance of algorithms, as proposed in:
> I. Rodríguez-Fdez, A. Canosa, M. Mucientes, A. Bugarín, STAC: a web platform for the comparison of algorithms using statistical tests, in: Proceedings of the 2015 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), 2015. 