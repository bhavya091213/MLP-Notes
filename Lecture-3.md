Lecture-3.md


# Lecture 3 - Linear Algebra in Machine Learning
- Why?
  - Data representation/transformation in a vector space (high dimensional)
  - Allows us to represent complex multi-dimensional data
  - Understand geometric shape
    - Decorrelation: whitening and data compression
      - PCA [Principal Component Analysis is a dimensionality reduction technique that transforms data into a new coordinate system such that the greatest variance by any projection of the data comes to lie on the first coordinate (the first principal component), and the second greatest variance on the second coordinate, and so on. The process is called decorrelation because it results in uncorrelated variables.]
    - PCA
  - Linear regression problems can be translated into solving a linear system.
    - For a system of linear equations $Ax = b$, where $A$ is a matrix, $x$ is a vector of unknowns, and $b$ is a vector of constants, we can solve for $x$ using techniques like Gaussian elimination, matrix inversion (if $A$ is invertible), or iterative methods.
  - Side note: Important to know the most significant axes.
    - Why?
- From data processing to modeling and learning, we need linear algebra [Linear algebra is a branch of mathematics concerned with the study of vector spaces and linear mappings between such spaces.] to help and play a role.


### Linear regression

Linear Regression Model: $aX_1 + bX_2 + cX_3 = Y$

Below is a representation of the data that could be used in a linear regression model, showing sample values for the independent variables ($X_1, X_2, X_3$) and the dependent variable ($Y$).
| $X_1$ | $X_2$ | $X_3$ |
|---|---|---|
| 1.2 | 3.4 | 5.6 |
| 2.3 | 4.5 | 6.7 |
| 3.4 | 5.6 | 7.8 |
| 4.5 | 6.7 | 8.9 |
| 5.6 | 7.8 | 9.0 |
| 6.7 | 8.9 | 10.1 |
| 7.8 | 9.0 | 11.2 |
| 8.9 | 10.1 | 12.3 |
| 9.0 | 11.2 | 13.4 |
| 10.1 | 12.3 | 14.5 |
The equation $aX_1 + bX_2 + cX_3 = Y$ represents a linear combination of independent variables ($X_1, X_2, X_3$), each multiplied by a coefficient ($a, b, c$ respectively), which results in a dependent variable ($Y$).
In matrix form, this can be represented as:
$\begin{bmatrix} X_{11} & X_{12} & X_{13} \\ X_{21} & X_{22} & X_{23} \\ \vdots & \vdots & \vdots \\ X_{n1} & X_{n2} & X_{n3} \end{bmatrix} \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{bmatrix}$
Where:
*   The first matrix contains the data for the independent variables, with each row representing an observation and each column representing a variable ($X_1, X_2, X_3$).
*   The second matrix is a column vector containing the coefficients of the linear regression model.
*   The third matrix is a column vector containing the corresponding values of the dependent variable ($Y$).
If we consider a specific set of coefficients, for example, $a=2, b=3, c=4$, and use the first row of the data table above ($X_1=1.2, X_2=3.4, X_3=5.6$), the calculation for $Y$ would be:
$Y = (2 \times 1.2) + (3 \times 3.4) + (4 \times 5.6)$
$Y = 2.4 + 10.2 + 22.4$
$Y = 35.0$
This process can be repeated for each row of data to predict the value of $Y$. The goal of linear regression is to find the coefficients ($a, b, c$) that best fit the observed data, minimizing the difference between the predicted $Y$ values and the actual $Y$ values.
and vectory multiply with a 1 x 3 vector a b c 
to yeild another table Y


A vector space R^N is a set of vectors that is closed by linear combinations.

*Vector addition and multiplication by a real number*
The vector addition and multiplication by a real number follows eight rules.
*   For any vectors **u**, **v**, and **w** in the vector space V, and any scalars c and d:
    *   **u** + **v** = **v** + **u** (Commutative property of addition)
    *   (**u** + **v**) + **w** = **u** + (**v** + **w**) (Associative property of addition)
    *   There exists a zero vector **0** such that **u** + **0** = **u** for all **u** in V. (Additive identity)
    *   For every vector **u** in V, there exists an additive inverse -**u** such that **u** + (-**u**) = **0**. (Additive inverse)
    *   c(**u** + **v**) = c**u** + c**v** (Distributive property of scalar multiplication over vector addition)
    *   (c + d)**u** = c**u** + d**u** (Distributive property of scalar multiplication over scalar addition)
    *   c(d**u**) = (cd)**u** (Associative property of scalar multiplication)
    *   1**u** = **u** (Multiplicative identity)
Vector space is closed under linear combination.
(This means that if you take any vectors from the vector space and perform scalar multiplication and addition on them, the resulting vector will also be in that same vector space.)
Questions:
1.  What is the simplest vector space?
    *   The simplest vector space is the zero vector space, which contains only the zero vector {**0**}.
2.  If V = {[0],[1]}, is this a vector space?
    *   No. To be a vector space, V must contain the zero vector and be closed under scalar multiplication and vector addition. In this case, the set contains the vector [1], but if we scale it by 0, we get [0]. However, we cannot guarantee closure with only these two elements without knowing if they represent components of a larger, consistently defined set. If V refers to the set {[0]} and {[1]} as distinct entities, it fails as a vector space because it likely lacks closure under addition and scalar multiplication (e.g., [1] + [1] = [2], which may not be in V, or 2 * [1] = [2]). If V = {[0], [1]} represents a set of 1-dimensional vectors, it is also not a vector space because it is not closed under addition (e.g., [1] + [1] = [2] which is not in V) and scalar multiplication (e.g., 2 * [1] = [2] which is not in V).


Subspace of a vector space is a subset that satisfies the requirements for a vector space.
Examples:
- Any line through (0,0,0) is in $\mathbb{R}^3$ [The set of all ordered triples of real numbers].
- Any plane through (0,0,0) is in $\mathbb{R}^3$.
- A line that does not pass through (0,0,0) is not a subspace of $\mathbb{R}^3$.
- A line does not pass through (0,0,0) in r^3???

Column space contains all linear combinations of the column vectors of a matrix A.


        Example:

        - Describe column/row space of the matrix A
        A =
        [
        1 0 1
        0 1 1
        ]

        1. Column space is R^2.
           * The column space of A is the set of all possible linear combinations of its column vectors.
           * The column vectors are [1, 0] and [0, 1].
           * Any vector in R^2 can be expressed as a linear combination of [1, 0] and [0, 1].
  
        2. Row space is a plane spanned by [1 0 1] and [0 1 1].
           * The row space of A is the set of all possible linear combinations of its row vectors.
           * The row vectors are [1, 0, 1] and [0, 1, 1].
           * A vector in the row space can be represented as $c_1[1, 0, 1] + c_2[0, 1, 1]$, where $c_1$ and $c_2$ are scalars.
           * This results in a vector of the form $[c_1, c_2, c_1+c_2]$.
           * Alternatively, any vector $(x, y, z)$ in the row space satisfies the equation $x + y - z = 0$.
              * To derive this: Let the vector be $v = c_1[1, 0, 1] + c_2[0, 1, 1] = [c_1, c_2, c_1+c_2]$.
              * Let $x = c_1$, $y = c_2$, and $z = c_1+c_2$.
              * Substituting $c_1$ and $c_2$ into the third component, we get $z = x + y$, which can be rearranged to $x + y - z = 0$.
           * This equation defines a plane in R^3.
           * Therefore, the row space is a subspace in R^3.
   1. x+y-z = 0
   2. a subspace in r^3


# Null Space and Row Space
## Key Ideas
- **Null space**: The set of all vectors **x** such that
  \[
  A\mathbf{x} = \mathbf{0}
  \]
- The **null space is orthogonal to the row space of \(A\)**.
- **Orthogonal** means: for any vector \(\mathbf{v}\) in the null space and any vector \(\mathbf{w}\) in the row space,
  \mathbf{v} \cdot \mathbf{w} = 0
---
## Example 1 (Incorrect Orthogonality Check — Needs Correction)
\[
A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
\]
- **Row space**: spanned by \((1,2)\) and \((3,4)\).
- Solve \(A\mathbf{x} = 0\):
  \begin{cases}
  x_1 + 2x_2 = 0 \\
  3x_1 + 4x_2 = 0
  \end{cases}
  From the first equation: \(x_1 = -2x_2\). Let \(x_2 = t\).
  \mathbf{x} = t \begin{pmatrix} -2 \\ 1 \end{pmatrix}
- Null space is spanned by \(\begin{pmatrix} -2 \\ 1 \end{pmatrix}\).
- Check orthogonality:
  (1,2) \cdot (-2,1) = -2 + 2 = 0
  (3,4) \cdot (-2,1) = -6 + 4 = -2 \neq 0
This shows that the initial claim was mistaken — not all row vectors are orthogonal to the null space here because the row vectors are not independent (the row space is actually **1D**). We must re-check with a clearer example.
## Example 2 (Correct Orthogonality)
A = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix}
- **Row space**: spanned by \((1,1)\).
  x_1 + x_2 = 0
  Let \(x_2 = t\), then \(x_1 = -t\).
  \mathbf{x} = t \begin{pmatrix} -1 \\ 1 \end{pmatrix}
- Null space is spanned by \(\begin{pmatrix} -1 \\ 1 \end{pmatrix}\).
  (1,1) \cdot (-1,1) = -1 + 1 = 0
✔ Null space is orthogonal to the row space.
## Edge Cases
- **Zero Matrix**
  - Example: \(A = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}\)
  - Row space = \(\{ \mathbf{0} \}\).
  - Null space = all of \(\mathbb{R}^2\).
  - Orthogonality holds since the zero vector is orthogonal to all vectors.
- **Identity Matrix**
  - Example: \(A = I_2\).
  - Row space = \(\mathbb{R}^2\).
  - Null space = only the zero vector.
  - Orthogonality holds since \(\mathbf{0}\) is orthogonal to everything.
- **Rectangular Matrices**
  - For an \(m \times n\) matrix, the row space is a subspace of \(\mathbb{R}^n\).
  - Null space is also a subspace of \(\mathbb{R}^n\).
  - The relationship still holds.
## Rank-Nullity Theorem
For an \(m \times n\) matrix \(A\):
\text{rank}(A) + \text{nullity}(A) = n
- \(\text{rank}(A)\): dimension of the row space (and column space).
- \(\text{nullity}(A)\): dimension of the null space.
- \(n\): the number of columns of \(A\).
So:
\dim(\text{Row Space}) + \dim(\text{Null Space}) = N, \quad \text{where row space is in } \mathbb{R}^N
  - For an \(m \times n\) matrix, the row space is a subspace of \(\mathbb{R}^n\).  
  - Null space is also a subspace of \(\mathbb{R}^n\).  
  - The relationship still holds.

---

## Rank-Nullity Theorem
For an \(m \times n\) matrix \(A\):  
\[
\text{rank}(A) + \text{nullity}(A) = n
\]  

- \(\text{rank}(A)\): dimension of the row space (and column space).  
- \(\text{nullity}(A)\): dimension of the null space.  
- \(n\): the number of columns of \(A\).  

So:  
\[
\dim(\text{Row Space}) + \dim(\text{Null Space}) = N, \quad \text{where row space is in } \mathbb{R}^N
\]


### Linear algebra in ML

Solving linear system Ax=b

A solution/multiple solutions of a linear system Ax=b exist if the vector b is in the column space of A.
*   **Single solution:**
    *   Main motivation question: How can we make this equation have a solution? How many solutions exist?
    *   If b isn't in the column space, there is no x and y such that x * c1 + y * c2 = b.
        *   [This implies that a linear combination of the column vectors of A cannot produce the vector b.]
*   **Multiple solutions:**
    *   Infinite many possibilities x y z if b is in the column space and the column vectors a1 a2 and a3 are not independent.
        *   [This means that if the column vectors are linearly dependent, there exist non-trivial coefficients that sum to the zero vector. This redundancy allows for infinitely many solutions when b is in the column space.]
*   **No Solution**
    *   No solution of a linear system (Ax=b) exists if the vector b is **not** in the column space of A.
        *   (The column space of a matrix A, denoted as Col(A), is the set of all possible linear combinations of the column vectors of A. If 'b' cannot be formed by any combination of A's columns, then there is no 'x' that satisfies Ax=b.)
*   In linear regression, we aim to solve [a,b,c]
    *   (In the context of linear regression, [a,b,c] likely represents the coefficients or parameters of a model. For instance, if fitting a quadratic model $y = ax^2 + bx + c$, these would be the coefficients 'a', 'b', and 'c'.)

### Projection and Eigenvalues and Orthogonality
- Projecting a vector
    *   Why do we do it?
        *   (To find the closest point in a subspace to a given vector, which is crucial for approximation and dimensionality reduction.)
    *   What value and data can we gain?
        *   (We gain information about the components of a vector that lie within a specific subspace, allowing us to understand how much of a vector can be "explained" by a set of basis vectors.)
    *   Give a simple example
        *   (Consider projecting vector $\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ onto the line spanned by vector $\mathbf{u} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$. The projection of $\mathbf{v}$ onto $\mathbf{u}$ is given by the formula:
            $$ \text{proj}_{\mathbf{u}} \mathbf{v} = \frac{\mathbf{v} \cdot \mathbf{u}}{\|\mathbf{u}\|^2} \mathbf{u} $$
            Here, $\mathbf{v} \cdot \mathbf{u} = (3)(1) + (4)(1) = 7$ and $\|\mathbf{u}\|^2 = 1^2 + 1^2 = 2$.
            So, the projection is:
            $$ \text{proj}_{\mathbf{u}} \mathbf{v} = \frac{7}{2} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 3.5 \\ 3.5 \end{bmatrix} $$
            This means that the closest point on the line spanned by $\mathbf{u}$ to the vector $\mathbf{v}$ is $\begin{bmatrix} 3.5 \\ 3.5 \end{bmatrix}$.)
- Eigenvalues
    *   Explain Eigenvalues
        *   (Eigenvalues ($\lambda$) are scalar values associated with a linear transformation (represented by a matrix) that describe how much a corresponding eigenvector is stretched or compressed by that transformation. They are the roots of the characteristic polynomial of the matrix.)
        *   (For a square matrix A, an eigenvalue $\lambda$ satisfies the equation $A\mathbf{v} = \lambda\mathbf{v}$, where $\mathbf{v}$ is a non-zero vector.)
    *   Explain Eigenvectors
        *   (Eigenvectors ($\mathbf{v}$) are non-zero vectors that, when a linear transformation is applied to them, only change by a scalar factor (the eigenvalue). Their direction remains unchanged.)
        *   (In the equation $A\mathbf{v} = \lambda\mathbf{v}$, $\mathbf{v}$ is the eigenvector corresponding to the eigenvalue $\lambda$.)
- Orthogonality
    *   How does it play into eigenvalues? Explain in depth
        *   (Orthogonality plays a significant role in the properties and applications of eigenvalues and eigenvectors, particularly for symmetric matrices.)
        *   (For a symmetric matrix A (where $A = A^T$), its eigenvectors corresponding to distinct eigenvalues are orthogonal.)
        *   (If a symmetric matrix has repeated eigenvalues, it is still possible to find a set of orthogonal eigenvectors that span the eigenspace.)
        *   (This property of orthogonal eigenvectors is extremely useful. For example, in spectral decomposition or Principal Component Analysis (PCA), we often work with symmetric matrices (like covariance matrices). The orthogonal eigenvectors of these matrices represent directions of maximum variance in the data, and they are mutually independent, which simplifies analysis and interpretation.)
        *   (The spectral theorem for real symmetric matrices states that such a matrix can be diagonalized by an orthogonal matrix. That is, if A is a real symmetric matrix, then $A = PDP^T$, where P is an orthogonal matrix whose columns are the orthonormal eigenvectors of A, and D is a diagonal matrix whose diagonal entries are the corresponding eigenvalues.)
        *   (Mathematically, if A is symmetric and $\mathbf{v}_1, \mathbf{v}_2$ are eigenvectors corresponding to distinct eigenvalues $\lambda_1, \lambda_2$, then $\mathbf{v}_1 \cdot \mathbf{v}_2 = 0$.)
            *   (Proof sketch:
                Assume $A\mathbf{v}_1 = \lambda_1\mathbf{v}_1$ and $A\mathbf{v}_2 = \lambda_2\mathbf{v}_2$, with $\lambda_1 \neq \lambda_2$.
                Consider $\mathbf{v}_1^T A \mathbf{v}_2$.
                We can write this as $\mathbf{v}_1^T (\lambda_2 \mathbf{v}_2) = \lambda_2 (\mathbf{v}_1^T \mathbf{v}_2)$.
                Alternatively, since A is symmetric, $A^T = A$.
                So, $(A\mathbf{v}_1)^T \mathbf{v}_2 = (\lambda_1 \mathbf{v}_1)^T \mathbf{v}_2 = \lambda_1 (\mathbf{v}_1^T \mathbf{v}_2)$.
                Also, $(A\mathbf{v}_1)^T \mathbf{v}_2 = \mathbf{v}_1^T A^T \mathbf{v}_2 = \mathbf{v}_1^T A \mathbf{v}_2 = \mathbf{v}_1^T (\lambda_2 \mathbf{v}_2) = \lambda_2 (\mathbf{v}_1^T \mathbf{v}_2)$.
                Therefore, $\lambda_1 (\mathbf{v}_1^T \mathbf{v}_2) = \lambda_2 (\mathbf{v}_1^T \mathbf{v}_2)$.
                $(\lambda_1 - \lambda_2) (\mathbf{v}_1^T \mathbf{v}_2) = 0$.
                Since $\lambda_1 \neq \lambda_2$, we must have $\mathbf{v}_1^T \mathbf{v}_2 = 0$, which means $\mathbf{v}_1$ and $\mathbf{v}_2$ are orthogonal.)



### Spectral Decomposition of a Symmetric Matrix (Geometric Shape)
- Full, Diagonal, and Spherical
- Can we guess the eigenvector and eigenvalue form [of] Gaussian contours?
- Spectral Decomposition of Symmetric Matrix (Covariance)
- Whitening (decorrelation)
- PCA (Principal Component Analysis)
### Data Whitening
- Mean = 0 and Covariance = I (Identity matrix)
$E = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}$, $\mu = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$
- How would we approach decorrelation?
Affine transformation $W = AX + b$ where $E[W] = 0$ and $COV(W) = I$
    - We can shift and rotate and scale.
    - (To achieve $E[W] = 0$, the mean of $X$ should be $\mu$. If $X$ has mean $\mu$, then $E[W] = E[AX + b] = AE[X] + b = A\mu + b$. For $E[W] = 0$, we set $b = -A\mu$. This implies the shift operation involves subtracting the mean.)
    - (To achieve $COV(W) = I$, we need to consider the relationship between $COV(W)$ and $COV(X)$. $COV(W) = COV(AX + b) = A COV(X) A^T$. We want to find a matrix $A$ such that $A COV(X) A^T = I$.)
    - (A common method for finding $A$ involves using the spectral decomposition of the covariance matrix $COV(X) = V \Lambda V^T$, where $V$ is the matrix of eigenvectors and $\Lambda$ is the diagonal matrix of eigenvalues. If we choose $A = \Lambda^{-1/2} V^T$, then $A COV(X) A^T = (\Lambda^{-1/2} V^T) (V \Lambda V^T) (\Lambda^{-1/2} V^T)^T = \Lambda^{-1/2} V^T V \Lambda V^T V (\Lambda^{-1/2})^T = \Lambda^{-1/2} I \Lambda I \Lambda^{-1/2} = \Lambda^{-1/2} \Lambda \Lambda^{-1/2} = I$. This $A$ achieves the desired decorrelation and scaling.)
    - (Therefore, the full whitening transformation can be expressed as $W = \Lambda^{-1/2} V^T (X - \mu)$.)


### PCA
- Represnetation of data manfiolds in a high dimensional space
- helps clean data
Here's the corrected and enhanced text:

*   **Introduction to Machine Learning**
    Machine learning (ML) is a type of artificial intelligence (AI) that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values.
    *   **Definition:**
        *   Machine learning is a field of artificial intelligence that focuses on the development of computer systems that can learn from and make decisions based on data.
        *   This learning process is achieved through the use of algorithms that identify patterns and relationships within data, allowing the system to improve its performance on a specific task over time.
        *   Unlike traditional programming, where rules are explicitly defined by humans, machine learning systems learn these rules implicitly from the data they are exposed to.
    *   **How it Works:**
        *   The process generally involves feeding a large dataset into an algorithm that is designed to recognize patterns and make predictions.
        *   The algorithm then builds a "model" based on this data.
        *   This model can then be used to make predictions or decisions on new, unseen data.
        *   **Example:** Imagine you want to build a system that can predict whether an email is spam or not. You would feed the algorithm a dataset of emails that have already been labeled as "spam" or "not spam." The algorithm would then learn to identify features (like certain keywords, sender addresses, or formatting) that are common in spam emails. Once trained, the model can be used to classify new incoming emails.
    *   **Key Components:**
        *   **Data:** The foundation of machine learning. The quality, quantity, and relevance of data significantly impact the performance of the ML model.
            *   **Types of Data:**
                *   **Structured Data:** Organized in a tabular format (e.g., spreadsheets, databases).
                *   **Unstructured Data:** Lacks a predefined format (e.g., text documents, images, audio, video).
                *   **Semi-structured Data:** Has some organizational properties but not as rigid as structured data (e.g., JSON, XML files).
        *   **Algorithms:** The "brains" of the ML system. These are mathematical procedures or sets of rules that enable the system to learn from data.
            *   **Types of Algorithms:**
                *   **Supervised Learning:** Algorithms learn from labeled data, where both the input features and the desired output are known. The goal is to predict the output for new, unseen inputs.
                    *   **Examples:** Linear Regression, Logistic Regression, Support Vector Machines (SVM), Decision Trees, Random Forests, Neural Networks.
                    *   **Equation for Linear Regression:**
                        The linear regression model aims to find a linear relationship between an independent variable ($X$) and a dependent variable ($Y$) by fitting a line to the data points. The equation of the line is:
                        $Y = \beta_0 + \beta_1 X + \epsilon$
                        Where:
                        *   $Y$ is the dependent variable (what we want to predict).
                        *   $X$ is the independent variable (the predictor).
                        *   $\beta_0$ is the y-intercept (the value of $Y$ when $X$ is 0).
                        *   $\beta_1$ is the slope of the line (the change in $Y$ for a one-unit change in $X$).
                        *   $\epsilon$ is the error term, representing the difference between the observed and predicted values.
                        The goal of linear regression is to find the values of $\beta_0$ and $\beta_1$ that minimize the sum of squared errors between the actual $Y$ values and the predicted $Y$ values.
                *   **Unsupervised Learning:** Algorithms learn from unlabeled data, identifying patterns, structures, or relationships without explicit guidance.
                    *   **Examples:** K-Means Clustering, Hierarchical Clustering, Principal Component Analysis (PCA), Association Rule Mining.
                    *   **Equation for K-Means Clustering (Objective Function):**
                        K-Means aims to partition $n$ observations into $k$ clusters, where each observation belongs to the cluster with the nearest mean (cluster centroid). The objective is to minimize the within-cluster sum of squares (WCSS), also known as inertia:
                        $WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$
                        *   $k$ is the number of clusters.
                        *   $C_i$ is the set of data points in the $i$-th cluster.
                        *   $x$ is a data point in cluster $C_i$.
                        *   $\mu_i$ is the centroid (mean) of cluster $C_i$.
                        The algorithm iteratively assigns data points to clusters and recalculates centroids until the centroids no longer change significantly.
                *   **Reinforcement Learning:** Algorithms learn by interacting with an environment, receiving rewards or penalties based on their actions. The goal is to learn a policy that maximizes cumulative reward over time.
                    *   **Examples:** Q-learning, Deep Q Networks (DQN), Proximal Policy Optimization (PPO).
                    *   **Equation for Q-Learning (Bellman Equation):**
                        Q-learning is a model-free reinforcement learning algorithm that learns an action-value function, $Q(s, a)$, which represents the expected future reward of taking action $a$ in state $s$ and then following the optimal policy thereafter. The update rule is based on the Bellman equation:
                        $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)]$
                        *   $s_t$ is the current state.
                        *   $a_t$ is the action taken in state $s_t$.
                        *   $r_{t+1}$ is the reward received after taking action $a_t$ and transitioning to state $s_{t+1}$.
                        *   $s_{t+1}$ is the next state.
                        *   $\alpha$ is the learning rate (0 < $\alpha$ <= 1), which determines the extent to which new information overrides old information.
                        *   $\gamma$ is the discount factor (0 <= $\gamma$ < 1), which determines the importance of future rewards. The closer $\gamma$ is to 1, the more important future rewards are.
                        *   $\max_{a} Q(s_{t+1}, a)$ represents the maximum expected future reward from the next state $s_{t+1}$.
        *   **Model:** The output of the training process. It's a representation learned from data that can be used to make predictions.
        *   **Evaluation Metrics:** Used to assess the performance of an ML model.
            *   **Examples:** Accuracy, Precision, Recall, F1-Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE).
            *   **Equation for Accuracy:**
                Accuracy is a common metric for classification tasks, representing the proportion of correct predictions out of the total number of predictions.
                $Accuracy = \frac{True \ Positive \ + \ True \ Negative}{Total \ Number \ of \ Observations}$
                Where:
                *   True Positive (TP): The model correctly predicted a positive class.
                *   True Negative (TN): The model correctly predicted a negative class.
                *   False Positive (FP): The model incorrectly predicted a positive class (Type I error).
                *   False Negative (FN): The model incorrectly predicted a negative class (Type II error).
                *   Total Number of Observations = TP + TN + FP + FN.
            *   **Equation for Mean Squared Error (MSE):**
                MSE is a common metric for regression tasks, measuring the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
                $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
                *   $n$ is the number of observations.
                *   $y_i$ is the actual value of the $i$-th observation.
                *   $\hat{y}_i$ is the predicted value of the $i$-th observation.
                Lower MSE values indicate a better fit of the model to the data.
    *   **Applications of Machine Learning:**
        *   **Healthcare:** Disease diagnosis, drug discovery, personalized medicine.
            *   **Real-life Example:** ML algorithms can analyze medical images (like X-rays or MRIs) to detect anomalies that might be missed by the human eye, aiding in early diagnosis of diseases like cancer.
        *   **Finance:** Fraud detection, algorithmic trading, credit scoring.
            *   **Real-life Example:** Banks use ML to identify fraudulent transactions by analyzing patterns in spending behavior that deviate from the norm for a particular customer.
        *   **E-commerce:** Recommendation systems, personalized marketing, inventory management.
            *   **Real-life Example:** Online retailers like Amazon use ML to recommend products to customers based on their browsing history, past purchases, and the behavior of similar customers.
        *   **Transportation:** Autonomous vehicles, traffic prediction, route optimization.
            *   **Real-life Example:** Ride-sharing services like Uber and Lyft use ML to predict demand, optimize driver allocation, and estimate arrival times.
        *   **Natural Language Processing (NLP):** Machine translation, sentiment analysis, chatbots.
            *   **Real-life Example:** Virtual assistants like Siri and Alexa use NLP to understand and respond to voice commands, and services like Google Translate use ML for language translation.
        *   **Image and Speech Recognition:** Facial recognition, voice assistants, content moderation.
            *   **Real-life Example:** Social media platforms use ML for facial recognition to suggest tagging friends in photos and for content moderation to identify and remove inappropriate material.
    *   **Types of Machine Learning Problems:**
        *   **Classification:** Predicting a categorical label (e.g., spam/not spam, disease A/disease B).
        *   **Regression:** Predicting a continuous numerical value (e.g., house price, stock price).
        *   **Clustering:** Grouping similar data points together without prior labels.
        *   **Dimensionality Reduction:** Reducing the number of features in a dataset while retaining important information.
        *   **Anomaly Detection:** Identifying unusual patterns or outliers in data.
    *   **Ethical Considerations in Machine Learning:**
        *   **Bias:** ML models can perpetuate or even amplify biases present in the training data, leading to unfair or discriminatory outcomes.
            *   **Real-life Example:** If a hiring algorithm is trained on historical data where certain demographics were underrepresented in specific roles, it might unfairly penalize candidates from those demographics.
        *   **Privacy:** The use of large datasets raises concerns about data privacy and security.
        *   **Transparency and Explainability:** It can be difficult to understand how complex ML models arrive at their decisions, leading to a "black box" problem.
            *   **Real-life Example:** In critical applications like medical diagnosis, it's important to understand *why* an ML model made a particular prediction to build trust and allow for human oversight.
        *   **Job Displacement:** Automation powered by ML may lead to changes in the job market.



### Pseudoinverse

- Projection to the column space
    - The pseudoinverse can be understood as a way to project a vector onto the column space of a matrix. This is particularly useful when dealing with systems of linear equations that may not have an exact solution.
- Finding an approximated solution
    - When a system of linear equations $Ax = b$ has no exact solution, the pseudoinverse allows us to find a vector $x$ that minimizes the error, i.e., it finds the least squares solution. This is often expressed as:
        $$x = A^+ b$$
        where $A^+$ is the pseudoinverse of $A$.
- Explain
    - The pseudoinverse of a matrix $A$, denoted as $A^+$, is a generalization of the matrix inverse. For a matrix that is not square or is singular (non-invertible), the pseudoinverse provides a "best possible" inverse. It satisfies certain properties, such as $AA^+A = A$ and $A^+AA^+ = A^+$.
- examples
    - Consider a simple overdetermined system:
      $$
      \begin{pmatrix} 1 \\ 2 \end{pmatrix} x = \begin{pmatrix} 3 \\ 1 \end{pmatrix}
      This system has no exact solution. The pseudoinverse of $A = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$ can be calculated, and then used to find the least squares solution for $x$.
- provide motivation
    - The motivation for using the pseudoinverse arises in situations where direct inversion is impossible or undesirable:
        - **Overdetermined systems:** Systems with more equations than unknowns, where an exact solution might not exist.
        - **Underdetermined systems:** Systems with fewer equations than unknowns, which may have infinitely many solutions. The pseudoinverse selects a specific solution (e.g., the one with the minimum norm).
        - **Singular matrices:** Square matrices that do not have an inverse.
        - **Regularization:** In some applications, the pseudoinverse can be used as a form of regularization to stabilize solutions.
- analogies
    - **Closest point on a line:** Imagine you have a target point in space, and you want to find the closest point on a specific line to that target. The projection of the target point onto the line gives you this closest point. The pseudoinverse acts similarly by projecting a "target" vector onto the column space of the matrix.
    - **"Best Guess" Calculator:** If a calculator is asked to perform an operation it cannot do exactly (like dividing by zero, though a real calculator would give an error), a hypothetical "best guess" calculator might return a result that is as close as possible to what would be expected in a similar, solvable scenario. The pseudoinverse provides this "best guess" for matrix operations that are not perfectly invertible.