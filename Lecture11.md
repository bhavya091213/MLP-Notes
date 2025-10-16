Lecture11.md


# Lecture 111: SVM
### Recap
- Margin: The smallest distance between the decision boundary and any of the samples
- SVM aims to maximize the margins
- The distance `d` can be represented as:
    $$ d = \frac{|w \cdot x + b|}{\|w\|} $$
- Primary problem was that, however this is a convex and inequality constrained problem
  - How many constraints?
  - Subject to $t_n(w^t x_n + b) \geq 1$ for all $n$
- KKT [Karush-Kuhn-Tucker conditions, a set of conditions used to derive the optimal solution for constrained optimization problems]
  - Primal feasibility
  - Dual feasibility
  - Stationarity
  - Complementary slackness
- Lagrangian Functions
  - Auxiliary terms that make it unconstrained
  - How many Lagrangian parameters are involved?
  - $1 - t_n(w^t \cdot x_n + b) \leq 0$
  - Due to the high number, we can use the stationary condition, from which we can derive two conditions that can find the optimal $w^*$ parameters
  - $w^*$ is the summation of $\lambda_n t_n x_n$ (where $\lambda_n$ are the Lagrange multipliers)
- We can do a dual representation
  - Comes from Lagrangian functions; we set the optimal parameters.
  - The dual function is the lower bound of the primal objective function.
  - Relationship between dual and primal.
  - If we find a maximum point for the dual function, we can find the optimal $\lambda^*$
- In summation, this is a quadratic optimization problem that we need to solve.
  - If we find a maximum point for dual functions we can find the optimal Lambda*



### SVM Classifiers
- Our goal is now to solve this dual representation problem
- Dual solution
  - $\lambda_1, \lambda_2, ..., \lambda_N$
- By complementary slackness, the following is true
  - if $\lambda_i = 0$
    - $t_i(w^T x_i + b) \ge 1$ (data points on the correct side of the margin or outside it)
  - if $\lambda_i > 0$
    - $t_i(w^T x_i + b) = 1$ (data points on the margin)
- By shifting $w^Tx + b = 0$ (to -1 and 1), $0$ is the margin boundary, the others are class $\pm 1$ respectively
  - the corresponding data that is beyond the margin boundaries
- Not all data points will be involved in deciding $w^*$
- Only data points that are on the margin (within the margin boundaries) will be involved in order to generate the hyperplane
  - hyperplane: A flat, two-dimensional surface that extends infinitely in all directions within a higher-dimensional space. In SVM, it refers to the decision boundary itself.
  - decision boundary plane: The surface that separates different classes. In the context of SVM with linear kernels, the hyperplane *is* the decision boundary. For non-linear SVMs, the decision boundary in the original feature space might not be a simple plane.
  - Those are the support vectors that will be used to define the support vector machine

- If we use the dual problem, we only need to know the inner products; we don't need explicit feature design.
  - Advantages?
### Kernel Functions
- Mapping real values to the feature map and the inner product.
- Once we have raw data points, without knowing the $\Phi$ (map), we can compute the value of the inner product.
- Ex:
  - $x_1, x_2$ we design a 3-dimensional feature map: $1, x_1^2, x_2^2$.
  - We can use $k(x, x')$ and project the feature maps and then do the feature map.
- $k(x_1, x_s)$ is a kernel function
  - if the Gram matrix $K(\Phi(x), \Phi(x'))$ is positive semidefinite.
    - What is positive semidefinite?
    - All eigenvalues are non-negative.
    - In a Gram matrix, any type of vector inner product is greater than or equal to 0.
- Constructing new kernels for any constant $c \ge 0$?
- Or by multiplying the function $f(x)k_1(x,x')f(x')$
- Any function polynomial $q$ with non-negative coefficients.
- So on so forth.
### Gaussian Kernel
- $k(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)$
  - A mathematical function shaped like a bell curve, widely used in image processing for blurring and smoothing by averaging pixels with their neighbors, and in machine learning for tasks like Support Vector Machines (SVMs) and Gaussian processes.
- The feature vector of the Gaussian kernel implicitly has an infinite-dimensional space.
  - What does this mean?
- exp -||x=x'||^2 / 2\sigma^2

Objective for Maximum Margin
- In the original data space, the data points are not linearly separable, but the kernel trick implicitly transforms them to an infinite-dimensional feature space, so a maximum margin classifier can be found, making the feature space linearly separable.
- Gaussian kernel implicitly projects your data points to infinite dimensions, and then you can achieve linear separations in that feature space.
- When you get back to the data space, you can form these non-linear classifications.
- We can control the complexity using the value of sigma.
  - If we use a small sigma, the Gaussian kernel will be very sharp.
  - $\gamma = 1/\sigma^2$ (where $\gamma$ is the kernel parameter and $\sigma$ is the standard deviation of the Gaussian kernel)
  - Effect of gamma on the number of support vectors and the decision boundary.
  - With a small gamma, some representative samples become support vectors.
  - With a large gamma, almost every sample becomes a support vector.
  - Depending on gamma, the complexity varies.
  - Some of the support vectors become representative and they will group other data points; that's the motivation for changing the gamma.
  - So in SVMs, we can control complexity using the sigma values.
  - What are the sigma values and where do they come from? (Sigma values, often referred to as the kernel width or bandwidth, are hyperparameters that need to be tuned during the model training process. They are typically not learned directly from the data but are selected through methods like cross-validation to optimize model performance.)
- Gaussian kernel embeds an infinite-dimensional feature space; is that high complexity okay with a finite number of data points? (This is a trade-off. While the kernel maps to infinite dimensions, the decision boundary in the original space can still be complex. The choice of kernel parameter influences how complex this boundary can become, and regularization terms in SVMs help prevent overfitting despite this high-dimensional mapping.)

### Parametric model vs. SVM
- Fixed number of parameters.
  - $y = w_1x_1 + ... + w_nx_n$ (This represents a linear model where y is the output, $w_i$ are the weights for each feature $x_i$)
  - High variance between train and test performance.
  - Overfitting.
  - All data points are involved in defining the model.
- In SVM:
  - Non-parametric.
  - One hyperplane defined by support vectors only, not all data points.
  - Not all data points are involved in defining the hyperplane.
  - If we have data points $S_1$ and $S_2$, even though their values could be different, but if they share the same support vectors (when we apply SVM to each dataset), then we can get the same hyperplane.
  - We still, at the same time, need to consider the original data set.
    - If the original data set is very entangled,
    - Complexity.
    - We can find optimal sigma through cross validation
* We can fold dataset and train on each fold in order to tune hyperparameters like sigma.
* Instead of hard margins, we can have soft margins.
    * Relaxation of the minimum margin.
    * We can allow a few data points to cross the margin.
    * $1 - \xi_n$ and $\xi_n \geq 0$ in $\mathcal{A}_n$ [$\mathcal{A}_n$ refers to the set of Lagrange multipliers associated with inequality constraints.]
    * We don't want to relax too much (we only relax to account for noise because we aren't going to have perfectly clean and binary data).
    * We introduce slack variables ($\xi_n$) and then we minimize.
        * $t_n(w^Tx_n + b) \geq 1 - \xi_n$ [This inequality defines the condition for a data point $x_n$ to be correctly classified, allowing for some slack $\xi_n$. $t_n$ is the label (+1 or -1), $w$ is the weight vector, and $b$ is the bias.]
* Lagrangian parameters:
    * New parameter $\mu$ [This is a Lagrange multiplier for the non-negativity constraint on slack variables.]
    * Corresponds to non-negativity constraint.
    * Stationarity.
        * $C - \lambda_n^* - \mu_n^* = 0$ [This equation arises from taking the partial derivative of the Lagrangian with respect to $\mu_n^*$ and setting it to zero, as part of the KKT conditions. $\lambda_n^*$ and $\mu_n^*$ are the optimal Lagrange multipliers.]
        * Stationarity condition is the reason the dual variables are "capped"/upperbounded by $C$, and, together with complementarity, it tells you exactly which training points are non-SVs (support vectors), margin SVs, or error SVs.
* Suppose we find the dual solutions.
* If one of our lambdas are 0, the constraints are an inequality by complementary slackness.
    * $\lambda_n + \mu_n = C$, that means $\mu_n = C$, that means $\xi_n = 0$ [This scenario describes data points that are correctly classified and lie on or beyond the margin.]
    * That means the data points are on the correct side but beyond the margin.
* The second case is that the lambda is between 0 and $C$.
    * The data points lie exactly on the margin.
* If $\lambda_n = C$:
    * $\mu_n = 0$, $\xi_n > 0$
    * Every data point is inside the margin.
* Reminder that the dual formation is to use the Lagrangian; that's the motivation.
  - The data points lie exactly on the margin
- If lambda = C
  - mu n = 0, \xi > 0
  - Every data point are inside he margin 
- Reminder that dual formation is to use the lagrangian thats the motivation
- She might give a set of lambdas and we gotta figure out where the data would be, within margin or beyond or eon the margin (quesiton on the midterm)
- If we find an optimal solution and corresponding hyperplans umst satisfy the three conditions
- Iteratively find lambda, intermediate process we could check every data points is satisfying one of these three conditions. Can decide algorithm is acheived an optimal solution or not. Extremely important conditions (test convergence of SMO algorithm)



### Complexity control by C
Below is a bunch of true false
- Large C value results in overfitting? then that is hard svm which is overfitting, so True, not underfitting
- Linearly sepeerable soft margin and hard margin will result in the same calssifier (soft margins will include any outliers, so won't exactly produce the same classifier as the hard margin)
- When data is not linearly seperable then o way for a hard margin calssifier converges F We can use gaussian kernels
- A hard marign SVM is sensitive to outliers F
- A soft margin SVm is sensitive to Outliers 
- if C is too large, there is chance algorithm may not converge This is True
  - No relaxation, f we don't use gaussian kernel or something then data cant't be linearly seperable and no convergence would happen


### One data point Loss Comparrson
- Soft VM, perceptron, and logistic regression
- Loss computation for oneData point
- [Image]("IMG_9448.jpg" "Image")
- Hinge Loss
- Once data poitns are beyond the
- SVM will not include all of data points in their predictions, only a subset and support vectors would be in part with the predctions




### Sequetintial Minitmal Optimization (SMO)
- SMO uses coordinate scent algorithm (iterative)
- Update one coordinte at a time
- Will iteratively find the optimum
- Cannot choose one coordinate
- 