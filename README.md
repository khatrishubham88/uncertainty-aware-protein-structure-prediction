# Introduction
In this work, we use AlphaFold as benchmark model. However, we only focus on predicting the pairwise distances of amino acids using both primary features which are the sequences of proteins themselves and
evolutionary features extracted from Multiple Sequence Alignments. Our model, similar to AlphaFold, predicts the distograms which are discretized distances. We only consider distances between 2 and 22 Angstroms.
Herefore, we use a deep two-dimensional dilated convolutional network with a variable number of residual block groups. 
Besides the accuracy of the model, it is often important to know how confident a neural network is in its prediction. However, modern neural networks tend to be poorly calibrated compared to the ones from a decade ago.
Therefore, we use two uncertainty quantification methods to raise the uncertainty awareness of the model.

# Insights of ProteinNet Data
* Contains Train/Test/Validation Splits
* Number at the end of Filename is the amount of thinning on the dataset.
* Data parser available in the ProteinNet Repository.
* Tertiary Features are the Labels
* Link to the [Repository](https://github.com/aqlaboratory/proteinnet); [Reference Paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2932-0)


# Model Repos
* [AlphaFold](https://github.com/deepmind/deepmind-research/tree/7bb484fffa87d3486ac791bb98b5b3dd65d8264e/alphafold_casp13); [Reference Paper](https://www.nature.com/articles/s41586-019-1923-7.epdf?author_access_token=Z_KaZKDqtKzbE7Wd5HtwI9RgN0jAjWel9jnR3ZoTv0MCcgAwHMgRx9mvLjNQdB2TlQQaa7l420UCtGo8vYQ39gg8lFWR9mAZtvsN_1PrccXfIbc6e-tGSgazNL_XdtQzn1PHfy21qdcxV7Pw-k3htw%3D%3D)
* [ProSPR](https://github.com/dellacortelab/prospr); [Reference Paper](https://www.biorxiv.org/content/10.1101/830273v2.full.pdf)
* [AlphaFold Pytorch](https://github.com/Urinx/alphafold_pytorch)
* [Open Protein](https://github.com/biolib/openprotein)

# Input Pipeline Repo
* [Minifold](https://github.com/EricAlcaide/MiniFold)
* [ProSPR](https://github.com/dellacortelab/prospr); [Reference Paper](https://www.biorxiv.org/content/10.1101/830273v2.full.pdf)
* [Open Protein](https://github.com/biolib/openprotein)
* [AlphaFold Pytorch (cropping here)](https://github.com/Urinx/alphafold_pytorch)

# UQ Paper
* [MC-Dropout, Gaussian dropout interpreted as performing Bayesian inference](https://arxiv.org/abs/1506.02142)
* [Temperature Scaling](https://arxiv.org/pdf/1706.04599.pdf)
* [Deep ensembles](https://arxiv.org/abs/1612.01474) 
* [https://papers.nips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift](Deep ensembles (https://arxiv.org/abs/1612.01474) 
https://papers.nips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift ) 

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

