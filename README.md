# Introduction
In this work, we use AlphaFold as baselin` model. However, we only focus on predicting the pairwise distances of amino acids using both primary features which are the sequences of proteins themselves and
evolutionary features extracted from Multiple Sequence Alignments. Our model, similar to AlphaFold, predicts the distograms which are discretized distances. We only consider distances between 2 and 22 Angstroms.
Herefore, we use a deep two-dimensional dilated convolutional network with a variable number of residual block groups. 
Besides the accuracy of the model, it is often important to know how confident a neural network is in its prediction. However, modern neural networks tend to be poorly calibrated compared to the ones from a decade ago.
Therefore, we use two uncertainty quantification methods to raise the uncertainty awareness of the model.

# Model Training
We are providing an easy to use training script called train.py with the following arguments:
* traindata_path: Path to training dataset.
* valdata_path: Path to validation dataset.
* classweight_path: Path to save model weights.
* val_thinning: Thinning of the validation set. [Choose between 30, 50, 70, 90, 95, 100]

An example of executing the training script train.py would be: python train.py --traindata_path "path/to/training-data" --valdata_path "path/to/val-data" --classweight_path "path/to/model-weights" --val_thinning 50

# Model Evaluation
We are providing an easy to use evaluation script named evaluate.py with the following arguments:
* testdata_path: Path to test dataset.
* model_path: Path to saved model weights.
* category: Category of the test set to use for evaluation. Choose between [1: TBM , 2: FM, 3: TBM-Hard, 4: TBM/TBM-Hard, 5: All]
* mc: Whether to use MC dropout.
* sampling: Number of samples to use for MC Dropout (MCD).
* ts: Whether to use Temprature Scaling (TS).
* temperature_path: In case TS flag is set, path to Numpy binary containing learned temperature.
* plot: Whether to plot evaluation set

An example of executing the evaluation script evaluate.py would be
* without MC dropout or TS: python evaluate.py --testdata_path "path/to/test-set" --model_path "path/to/model-weights" --category 5
* with TS: python evaluate.py --testdata_path "path/to/training-data" --model_path "path/to/model-weights" --category 5 --ts --temperature_path "path/to/temperature"
* with MC dropout: python evaluate.py --testdata_path "path/to/test-set" --model_path "path/to/model-weights" --category 5 --mc --sampling 100

# Temperature Scaling
We are providing an executable file called temperature_scaling.py to learn the temperature on the validation set for a given model. 
This program will create a folder called temperatures and save the temperature under temperature.npy.
It uses following arguments:
* traindata_path: Path to training set.
* valdata_path: Path to validation set.
* modelweight_path: Path to saved model weights.
* epochs: Number of sweeps through validation set to learn temperature e.g. 1.
* iterations: Number of optimization steps to use per batch e.g. 3.
* lr: Learning rate e.g. 0.001.

An example of executing the file temperature_scaling.py would be:
* python temperature_scaling.py --traindata_path "path/to/train-set" --valdata_path "path/to/val-set" --model_path "path/to/model-weights" --epochs 1 --iterations 3 --lr 0.001

# Related Repositories
* [AlphaFold](https://github.com/deepmind/deepmind-research/tree/7bb484fffa87d3486ac791bb98b5b3dd65d8264e/alphafold_casp13); [Reference Paper](https://www.nature.com/articles/s41586-019-1923-7.epdf?author_access_token=Z_KaZKDqtKzbE7Wd5HtwI9RgN0jAjWel9jnR3ZoTv0MCcgAwHMgRx9mvLjNQdB2TlQQaa7l420UCtGo8vYQ39gg8lFWR9mAZtvsN_1PrccXfIbc6e-tGSgazNL_XdtQzn1PHfy21qdcxV7Pw-k3htw%3D%3D)
* [ProSPR](https://github.com/dellacortelab/prospr); [Reference Paper](https://www.biorxiv.org/content/10.1101/830273v2.full.pdf)
* [AlphaFold Pytorch](https://github.com/Urinx/alphafold_pytorch)
* [Open Protein](https://github.com/biolib/openprotein)
* [Minifold](https://github.com/EricAlcaide/MiniFold)


# UQ Paper
* [MC-Dropout, Gaussian dropout interpreted as performing Bayesian inference](https://arxiv.org/abs/1506.02142)
* [Temperature Scaling](https://arxiv.org/pdf/1706.04599.pdf)
* [Deep ensembles](https://arxiv.org/abs/1612.01474) 
* [Can you trust your models](https://papers.nips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift)

# Libraries
* [Keras](https://keras.io/getting-started/functional-api-guide/)
* [Tensorflow](https://www.tensorflow.org/tutorials/quickstart/beginner)
* [pytorch](https://pytorch.org/tutorials/)

# Additional Resources 
* [Protein Folding](https://en.wikipedia.org/wiki/Protein_folding)
* [Multiple Sequence Alignment (MSA)](https://en.wikipedia.org/wiki/Multiple_sequence_alignment)
* CASP: [Wiki](https://en.wikipedia.org/wiki/CASP); [Other (Not sure how good)](http://predictioncenter.org/casp13/)
* [PDB](https://www.rcsb.org/)
* [AlphaFold Talk](https://www.youtube.com/watch?v=uQ1uVbrIv-Q)
* [AlphaFold presentation](http://predictioncenter.org/casp13/doc/presentations/Pred_CASP13-DeepLearning-AlphaFold-Senior.pdf)

