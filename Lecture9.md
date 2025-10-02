Lecture9.md

# Lecture 9 - Logistic Regression

Decision boundaries are the same as X1 + X2 = 0.

*   Important is where w is pointing to.
*   $\vec{w} \cdot \vec{x} + b = 0$
    *   $\vec{w}$: normal vector [vector representing weights]
        *   Direction to be the class 1 [indicates the direction in which the model classifies as class 1].
*   Direction and magnitude of the normal vector for the hyperplane determines orientation and steepness of the sigmoid function.
*   Steepness varies according to the magnitude of $||\vec{w}||$.
    *   The smoothest [decision boundary] will have a small magnitude, like (0.2, 0.2).
*   $w * = \underset{w}{\operatorname{argmax}} \mathcal{L}(D|w)$ (maximum likelihood)
    *   Bayes
    *   ensures $w$ is uniform
    *   if we have many data points, it's safe to use MLE
### Cross-entropy loss
- Cross-entropy loss
- $\sum_{n=1}^{N} (-t_{n} \ln \sigma(w^Tx_n) - (1-t_n)\ln(1-\sigma(w^Tx_n)))$
    *   ($\sigma$ represents the sigmoid function)
- Interpreted as probability
  - $P(t_n = 1 | X_n)$
  - $P(t_n = 0 | X_n)$
- Subtracting both for each data point
- We can minimize that function
  - Need a strategy
  - Need to figure out how the function looks
  - second derivative information at point $w_0$
    - Gives information about concavity
    - If $y = x^3$, second derivative is $y'' = 6x$. Inflection points will give info about concavity.
    - If $x > 0$, second derivative is positive, the converse is true.
  - Turning points: gradient = 0
  - Turning points gradient = 0
- Hessian
  - matrix is a square matrix of second-order partial derivatives of a scalar-valued function, or scalar field. It describes the local curvature of a function of many variables
* Data features are correlated
* If $||w||$ is increasing towards infinity, then 1 of them will be 0. This is true because, that means the outcomes will either be 1 or 0.
    * (This implies that if the norm of the weight vector $w$ becomes infinitely large, at least one of the components of $w$ must be zero. This scenario is related to regularization techniques where large weights are penalized, and in extreme cases, some weights might be completely zeroed out, leading to sparsity in the model.)
* Same $k$, objective functions will be the same.
* If data points are linearly correlated
    * In that case, we will have a constant flat surface.
    * Implies multiple local minima.
* When can we find one unique solution?
    * When data is not linearly separable.
    * Multiple $w$ are possible.
    * No minimum exists.
    * Infinitely many solutions, global minimum.
    * If the Hessian > 0 at $w$, unique solution.
        * (The Hessian matrix is the matrix of second-order partial derivatives of a scalar-valued function. If the Hessian evaluated at a critical point is positive definite, it indicates a local minimum. For the objective function $J(w)$, a positive definite Hessian at a critical point $w$ guarantees that this point is a unique local minimum. If this minimum is also the global minimum, then we have found a unique solution.)
    * Note that depending on the structure of the data, the shape of $J(w)$ varies.


### Steepest gradient descent algorithm
- iterative optimization method to minimize $J(\vec{W})$
Goes through and gradually finds optimal points
* $J(w)$ is a surface function on the domain of the parameters
  * We will look for $w^{(0)}$ (initial parameter vector), move by a predefined step size, compute directions, and move on from there
* We have a global minimum even though the local minimum exists
* $w^{(i+1)} = w^{(i)} - \eta \cdot \nabla J(w^{(i)})$
  * $-\nabla J(w^{(i)})$ is the steepest ascent direction (in this context, it points towards increasing the function value, so subtracting it from $w^{(i)}$ moves us in the direction of steepest *descent*)
  * $\eta$ (eta) is the step size (also known as learning rate)
  * ensure loss function is getting down smoothly
  * Each vector in the generated sequence has a lower cost than its predecessor (cost of height)
* Too large step size ($\eta$) is overshooting
  * slower convergence, keeps bouncing back and forth and wasting movements
  * might diverge
* Too small step size ($\eta$)
  * too slow to converge
  * might be stuck in a local minimum
* How to determine step size ($\eta$)?
### Training logistic regression using gradient steepest descent
- First need to compute gradients
- Computed based off data points
  - current w<sup>t</sup> \* current data point - ground truth all multiplied by x<sub>n</sub> (current data point)
  - Iteratively update w<sub>i+1</sub>, w<sub>i+2</sub> and so on so forth
- Observe how J(W) changes
  - Any stagnations or convergences, then can stop the algorithm
  - retrieves w\* which is your logistic regression model
- Gradients of objective functions will be very small
  - distance between ground truth and prediction is small
- Updates will be small, therefore as everything approaches 0 you know you are stopping
    * Fixed step size, automatically adjusted by movement since you multiply by the gradient. Local minimum gradients will be reduced along the way, so the products get smaller.
### Multinomial logistic regression
- Learning $K$ posterior $P(C_k|x)$
    - $K$ denotes the number of classes.
- "K # of classes"
- Takes all the weights, softmax function -> each of them then get identified to a class
    - The softmax function is used to convert a vector of raw scores (logits) into a probability distribution over the $K$ classes. For a given input $x$ and weights $w_k$ for each class $k$, the probability of belonging to class $k$ is calculated as:
        $$ P(C_k|x) = \frac{e^{w_k^T x}}{\sum_{j=1}^{K} e^{w_j^T x}} $$
    - This ensures that the probabilities for all classes sum to 1.
- Regularization gradient (add the prior)
    - When regularization is applied, a prior distribution over the model weights is incorporated. This can be thought of as adding a penalty term to the loss function. For example, with L2 regularization (also known as weight decay), the gradient of the regularization term with respect to the weights is added to the gradient of the loss function. If the regularization term is $\lambda \|w\|^2$, the gradient with respect to $w$ is $2\lambda w$.
- Takes all the weights, softmax function -> each of them then get identified to a class

### Perceptrons
- Rosenblatt, F. (1962). *Principles of Neurodynamics: Perceptrons and the Theory of Brain Mechanisms*. Washington, D.C.: Spartan Books.
- Step function with linear regression models
- Training
  - Objective function and optimization (update rule)
- Convergence theorem
  - Perceptron algorithm converges if data is linearly separable [meaning a line (or hyperplane in higher dimensions) can perfectly divide the data points of different classes.]
- Loss comparison between logistic regression vs perceptron
- Perceptron as a foundational element of Neural Nets

Generative:
* Gaussian Discriminant Analysis (GDA), Naive Bayes
* Discriminative: Logistic Regression
- Directly learns a decision boundary (hyperplane) for binary classification without considering any probabilistic modeling (no posterior interpretation) [This means it focuses on drawing a line to separate classes, rather than estimating the probability of belonging to a class.]
- The outer functions are step activation functions used to predict outcome class directly [A step function outputs a binary value (e.g., 0 or 1, -1 or 1) based on whether its input exceeds a threshold.]
- If right classification, the product between prediction and ground truth is positive
  - For prediction $ \hat{y} \in \{1, -1\} $ and ground truth $ y \in \{1, -1\} $:
    - If $ \hat{y} = y $, then $ \hat{y} \cdot y = 1 $ (positive)
    - If $ \hat{y} \neq y $, then $ \hat{y} \cdot y = -1 $ (negative)
  - This implies that if the classification is correct, the product $ \hat{y} \cdot y $ will be $ 1 $. If incorrect, it will be $ -1 $.
  - The text mentions "f(y) - 1 and f(y=-1)". This likely refers to the desired output or the sign of the product for correct classifications. If the prediction $\hat{y}$ is correct, and assuming the ground truth $y$ is either $1$ or $-1$, then $ \hat{y} = y $, and the product is $1$.  If we consider a prediction function $f(x)$ where $f(x)=1$ for one class and $f(x)=-1$ for another, then $f(y)$ is the predicted value for a data point with true label $y$. The condition for correct classification would be $f(y) = y$.
- THe outer functions are step activation functions used to predict outcome calss directly
- If right classification, the product between prediction and ground truth is positive
  f(y) - 1 and f(y=-1)

* Dendrites are similar to the inputs.
    * The cell bodies are each layer.
        * All the $w^T \cdot x$ are the parameters.
        * If $> 0$ it's activated; if $< 0$ it's not activated.
        * Objective function is piecewise linear, but differentiable at the current $w$.
          * Hyperplanes are changing over the domains of the $w$, therefore piecewise.
          * Differentiable because at the current local point, you can always compute the gradients for the objective functions.
        * Misclassification samples:
          * What are they?
            * Misclassification samples are data points that are incorrectly predicted by the model.
          * What is the error and how does it relate?
            * The error for a misclassified sample is the difference between the predicted output and the true label. This error contributes to the overall loss function, which the model aims to minimize. For instance, in a binary classification task where the true label is 1 and the model predicts 0, the error indicates a failure in the model's decision boundary.
            * Relationship to objective function: The objective function often incorporates a penalty for misclassifications. By minimizing the objective function, the model implicitly tries to reduce the number of misclassified samples.
- Can we gaurantee that it decreases the overall classification error?
