# strat-group-split
This repository contains code to perform stratified splitting 
of grouped datasets into train/validation sets or K-folds
using optimization.

## Summary
Given a labeled and grouped dataset, we want to split it into 
training and validation sets (or equally sized K folds)
while keeping the label 
distribution as close as possible on both and group integrity. 
After breaking the data into the two datasets, the groups must 
maintain their integrity, assigned to either set and not split 
among them. Furthermore, the splitting process should closely 
respect the imposed splitting proportion and label 
stratification.

The expected result for this problem is, given an input dataset, 
the list of groups assigned to each dataset, ensuring that both 
the train/validation split and the stratification are as close 
as possible to the specified values.

## Using the Code
### Train/Validation Split
All the code is contained in the `group_split.py` file.
The `main` function runs a benchmark between the two
optimization algorithms. It generates a problem matrix using
the `generate_counts` function and then submits it to both
algorithms, outputting the time taken, final cost value and
the approximations to both the desired split and the 
stratification.

Please note that the code is on a proof-of-concept stage. In 
the future I plan to create an independent Python package
with these ideas.

### K-Fold Split
All the code is contained in the `k_fold_split.py` file. You can 
alternatively use the `k-fold.ipynb` Jupyter notebook.

## Medium Articles
[Stratified Splitting of Grouped Datasets Using Optimization](https://towardsdatascience.com/stratified-splitting-of-grouped-datasets-using-optimization-bdc12fb6e691)

[Stratified K-Fold Cross-Validation on Grouped Datasets](https://towardsdatascience.com/stratified-k-fold-cross-validation-on-grouped-datasets-b3bca8f0f53e)