Lecture4.md

# Lecture 4 - Linear Regression
Predicting **y**, a real number that is a function of a bunch of variables
- Modeling
  - Linear combination of basis functions/features and parameters
  - Data preprocessing
- Learning
  - Maximum Likelihood Estimation (MLE)
  - Minimum Mean Square Error (MMSE) estimation
  - Normal equation and its solution
  - Pseudo-Inverse
### Spectral Decomposition
- A symmetric matrix can be represented by a matrix of eigenvectors and eigenvalues
- $e_1, e_2, \dots, e_n$ x matrix of diagonals being eigenvalues
### Solving Linear Equations
- Two cases for Ax = b
  - When b is in the column space of A
    - An exact solution exists.
  - When b is not in the column space
    - No solution
    - $A^T Ax = A^T b$
      - (This equation is derived by projecting the vector $b$ onto the column space of $A$, effectively finding the closest possible vector in the column space to $b$. The solution to this normal equation provides the coefficients that minimize the error.)
We can find an approximate solution when b is not in the column space
- Exact solution can be found if we are in the column space.
  - when b is not in the column space
    - No solution
  - If AtA is non singular where determinants arent 0 one unique solution invertile
  - if A^tA is singular and determinants are 0, infinte many soltions


### Linear Regression Problem
- Each feature dimension has semantic meaning.
- No semantic meaning, a whole vector represents something.
  - This problem will be our focus.
Dependent variable
- Regressand
- Response
- Endogenous
- Target
- Predicted
- Explained Variable
Independent Variable
- Regressor
- Covariate
- Exogenous
- Feature
- Predictor
- Explanatory variable
The main terms used will be targets and features.
- Predictor
- Explanatory variable

The main temrs used will be targets and featrues



### Preprocesing

- Centering
- Normalization
- Standardization
- Whiteneing



### MLE and MMSE
Frequentist assumes w parameter as fixed values and perform MLE to estimate the parameters.
MLE can be interpreeted as a special case of MAP when the prior density p(w) is uniform

Frequentist vs Bayes Estimation 
- w *= argmax P(D|w) MLE (Maximum likelihood estimation
- w *= argmaxp(w)D) = p(D|w)p(w) / p(D): Maximum A posteriori Estimation MAP

