# strat-group-split
Code to perform stratified split of grouped datasets into train 
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

