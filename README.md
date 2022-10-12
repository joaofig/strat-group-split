# strat-group-split
Code to perform a stratified split of grouped datasets into train 
and validation sets using optimization

## Summary
Given a labeled and grouped dataset, we want to split it into 
training and validation sets while keeping the label 
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
All the code is contained in the `group_split.py` file.
The `main` function runs a benchmark between the two
optimization algorithms. It generates a problem matrix using
the `generate_counts` function and then submits it to both
algorithms, outputting the time taken, final cost value and
the approximations to both the desired split and the 
stratification.

Please note that the code is on a proof-of-concept stage. In 
thee future I plan to create an independent Python package
with these ideas.

## Medium Article
[Stratified Splitting of Grouped Datasets Using Optimization](https://towardsdatascience.com/stratified-splitting-of-grouped-datasets-using-optimization-bdc12fb6e691)
