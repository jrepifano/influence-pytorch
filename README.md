# influence-pytorch
PyTorch implementation of Influence Functions.

Original method by ["Understanding Black-box Predictions via Influence Functions"](https://github.com/kohpangwei/influence-release)

## What's different from the original?
This repository contains the extended implementation to any vanilla PyTorch model.
Previously this was implemented by the original authors using the bottle-neck features
of a model into a logisitic regression model. While functional, this implementation left much to be
desired. This implementation computes gradients in an end-to-end manner to ensure the result 
is faithful to the original model.

### Files
- influence_pytorch.py: Implementation of equations 2 and 5 from the original paper
- test.py: trains a model and computes both influences for a random classification problem

### Dependencies
- PyTorch (tested with v1.7.1 but should work with older versions)
- NumPy
- Scipy/Scikit-learn (only necessary to run the test script)



If you have any questions please contact Jacob Epifano ([jrepifano@gmail.com](jrepifano@gmail.com))
