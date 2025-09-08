Lecture-2.md


# Lecture 2 - Probability for ML
### Why probability?
- We could predict dice # [number] if we knew every factor that affects the outcome
  - Grip
  - Angle
  - Air condition, etc [etcetera]
- We only have partial information
- Therefore, we need probability to guess the outcome of a scenario with the likelihood of the output
- Provides mathematical machinery to measure uncertainty associated in events
Probabilistic modeling targets to learn joint/conditional density
Modeling errors are considered as even for non-probabilistic/deterministic modeling, we need to consider random error *e*
Causes of random error (*y* = *f<sub>w</sub>(x)* + *e*: as we predict *y* based on *x*):
  1.  limited knowledge/accessibility to the features
  2.  limited hypothesis space/limited by #data points
  3.  measurement errors
  *   Here's the equation broken down:
        *   *y* is the predicted value.
        *   *f<sub>w</sub>(x)* represents a function that predicts *y* based on *x*, parameterized by *w* (weights or parameters).
        *   *e* represents the random error. This is the difference between the actual value of *y* and the value predicted by the function *f<sub>w</sub>(x)*.
  


### Frequentist vs Bayesian Probability
Let there be 4 balls, three red and one blue
The true probability is a soft number
Frequentist probability is by doing independent trials numerous times
- measuring relative beliefs [Corrected spelling]
- P = # of event A / # of trials
  *  (This is the basic frequentist probability calculation.)
Bayesian relies on a prior, it believes [Corrected spelling] something has happened
- P[event | evidence] proportional to P[evidence | event]P[event]
  * (This is Bayes' Theorem, often written as:)
     *   $$P(A|B) \propto P(B|A)P(A)$$
     *   where A is the event and B is the evidence.
- measuring uncertainty or belief [Corrected spelling]
Probability Space [Ω,2<sup>|Ω|</sup>,P]  [Using proper Omega symbol]
1. Experiment: [Corrected spelling]
   1. Any process of obtaining or generating an observation
      1. Inspection of an instance [Corrected spelling] item is defective or non-defective
2. Sample Space Ω: a set of all possible outcomes [Using proper Omega symbol]
   1. Ex:
      1. Ω = {non-defective, defective} [Using proper Omega symbol]
3. Events Set: (A subset of Ω or A ∈ 2<sup>|Ω|</sup>): a set of all possible subsets of Ω  [Using proper Omega symbol]
      1. 2<sup>|Ω|</sup> = {∅, {non-defective}, {defective}, Ω} [Using proper Omega symbol and showing null set symbol]
   2. Could also be called a power set
4. Probability Measure P[E]: a function P: 2<sup>|Ω|</sup>→[0,1]  [Using proper Omega symbol]
   1. E:
      1. P[{Defective}] = monitor assembly line for a period of time, compute the relative frequency
      1. 2^|Omega| = {Null Set, {non-defective}, {defective}, Omega}
   2. Could also be called a power set 
4. Probability Measure P[E]: a function P: 2^|Omega|->[0,1]
   1. E:
      1. P[{Defective}] = monitor assembly line for a period of time, compute the relative frequency

### Probability Axioms
- Probability Measure follows the three axioms
1. Non negativity
   1. P[A] >= 0
     *  (Probability of event A is greater than or equal to zero)
2. Total Probability
   1. P[Ω] = 1
     *  (Probability of the entire sample space Ω is equal to one)
3. Countable Additivity
   1. A<sub>i</sub> ∩ A<sub>j</sub> = ∅ if i != j => P[⋃<sub>k</sub>A<sub>k</sub>] = ∑<sub>k</sub> P[A<sub>k</sub>]
     * (If the intersection of A sub i and A sub j is null or empty if i is not equal to j, then the probability of the union of A sub k is equal to the Sum of the probability of A sub k)
     * (Events A<sub>i</sub> and A<sub>j</sub> are mutually exclusive or disjoint if their intersection is empty (∅))

**Corollary**: Probability of A + the A complement [or complement of A] = 1
  * (This is often represented as P(A) + P(A') = 1, or P(A) + P(A<sup>c</sup>) = 1, where A' or A<sup>c</sup> represents the complement of A)

### Random Variables
- To handle numerical outcomes/events
- Data samples and internal/output representations of ML system [Machine Learning system]
- E -> sample -> random variable
  - 3 Coins -> Sample space -> # of heads (example of RV)
- We can map the outcomes to individual numbers
- Let RVX be the indicator function for tail event
- Once RDV [Random Variable] is defined, need to define the CDF and PDF
  - Cumulative Distributive Function
    - Probability of the event X {X<=x}
    - F<sub>X</sub>(x) = P[X<=x] for -inf <= x <= +inf
      *   (This is the cumulative distribution function, representing the probability that the random variable X takes on a value less than or equal to x)
  - Probability Density Function
    - integral from x to infinity of (f<sub>X</sub>(dx))
      *   (The PDF is the derivative of the CDF.  The integral of the PDF over a range gives the probability of the random variable falling within that range.)
      *   (The PDF, denoted as f<sub>X</sub>(x), represents the probability density at each value x.)
      *   (The correct mathematical representation should be an integral from negative infinity to x:  ∫<sup>∞</sup><sub>x</sub> f<sub>X</sub>(t) dt = F<sub>X</sub>(x))
      *   P[a<=X<=b] = integral from a to b Fsubx(x)dx


### Computing Probability
Notes:
Three types
1. Conditional and Joint Probability
   1. As equally likely outcomes, P[A] becomes a counting problem
      *  $P(A) = \frac{|A|}{|S|}$ (Where |A| is the number of outcomes in event A, and |S| is the total number of outcomes in the sample space)
2. Independence and Conditional Independence
   1. Biased Outcomes
3. Marginalization and Partition
   1. Outcomes not equally likely
   2. Can be split into disjoint sets
      *   (Disjoint sets are sets that have no elements in common)
### Computing Conditional Probability
- Breast Cancer Example
  - Accuracy of test is usually between .9 and .95
  - Multiple cases:
    - If mammogram is correct or not
    - If there is breast cancer or not

Probability chain rule
1. P[ANB] = P[A] * P[B|A] = P[B] * P[A|B]
2. P[ANBNC] = P[A]*P[B|A]*P[C|ANB]

This is all just Bayes' Theorem:
$$P(A|B) = \frac{P(B|A) * P(A)}{P(B)}$$
(Where:
*  P(A|B) [is] the probability of event A occurring given that event B has already occurred
*  P(B|A) [is] the probability of event B occurring given that event A has already occurred
*  P(A) [is] the probability of event A occurring
*  P(B) [is] the probability of event B occurring)

Before going on vacation, ask your friend to water your plant
Without water, plant has 80 percent chance of dying
With watering, 20 percent chance of dying
Friend might forget to water it with a 30 percent chance
[Answer with the Bayesian formulas and define variables:]
Let:
* D = Plant is dead
* W = Plant is watered
* NW = Plant is not watered (friend forgot)
We are given:
* P(D|NW) = 0.8 [Probability plant is dead given it was not watered]
* P(D|W) = 0.2 [Probability plant is dead given it was watered]
* P(NW) = 0.3 [Probability friend forgot to water]
* P(W) = 0.7 [Probability friend watered (1 - P(NW))]
1. What is the chance that your plant will survive the week?
    We want to find P(not D) [P(¬D)], which is 1 - P(D) [1 - P(D)].  First, we need to find P(D) [Probability the plant is dead]:
    
    P(D) = P(D|W) * P(W) + P(D|NW) * P(NW) [Law of Total Probability]
    P(D) = (0.2 * 0.7) + (0.8 * 0.3) = 0.14 + 0.24 = 0.38
    P(not D) = 1 - P(D) = 1 - 0.38 = 0.62
    Therefore, there is a 62% chance the plant will survive the week.
2. If your friend forgot to water, what is the chance it will be dead when you return?
    This is already given: P(D|NW) = 0.8.
    Therefore, there is an 80% chance it will be dead.
3. If it is dead when you return, what is the chance your friend forgot to water it?
    We want to find P(NW|D) [Probability friend forgot to water given the plant is dead], using Bayes' Theorem:
    $$P(NW|D) = \frac{P(D|NW) * P(NW)}{P(D)}$$
    $$P(NW|D) = \frac{0.8 * 0.3}{0.38} = \frac{0.24}{0.38} \approx 0.6316$$
    Therefore, there is approximately a 63.16% chance your friend forgot to water it, given that the plant is dead.



### Expectations of Mean and Variance
$E[X] = \sum xP(x)$ (Expected value of a discrete random variable)
$E[X] = \int x f(x) \, dx$ (Expected value of a continuous random variable)
1.  Compute $E[X]$ if $P[X=1] = 1/3$ and $P[X=0] = 2/3$
   1.  $1 * 1/3 + 0 * 2/3 = 1/3$  (Calculation of expected value)
      *   $E[X] = (1 * 1/3) + (0 * 2/3) = 1/3$
2.  Suppose X is Bernoulli ($P[X=1]=1/3$) and Y is binomial $\binom{3}{y}$ [where y represents the number of successes] then $E[X]$ and $E[Y]$
    *   $E[X] = p$ (Expected value of a Bernoulli random variable, where p is the probability of success)
        *   In this case, $E[X] = 1/3$
    *   $E[Y] = np$ (Expected value of a Binomial random variable, where n is the number of trials and p is the probability of success on each trial)
        *   In this case, $E[Y] = 3 * 1/3 = 1$
   1. 1 *1/3 + 0*2/3 =1/3 
2. Suppose X is Bernoulli (P[X=1]=1/3) and Y is binomial 3 choose y then E[X] and E[Y]


### Variance: E[X^2] - (E[X])^2
*   Variance [A measure of statistical dispersion indicating the extent to which different possible values differ from the average value. For a random variable X, the variance is defined as:]
    $$Var(X) = E[X^2] - (E[X])^2$$
    *   Where:
        *   $E[X^2]$ represents the expected value of X squared.
        *   $(E[X])^2$ represents the square of the expected value of X.
-   Covariance [A measure of how much two random variables change together.]
    *   How the two RVs [random variables] linearly covary?
    *   The formula for covariance is:
        $$Cov(X, Y) = E[(X - E[X])(Y - E[Y])]$$
        *   Where:
            *   $Cov(X, Y)$ is the covariance between random variables X and Y.
            *   $E[X]$ is the expected value (mean) of X.
            *   $E[Y]$ is the expected value (mean) of Y.
            *   $E[(X - E[X])(Y - E[Y])]$ is the expected value of the product of the differences between each variable and its mean.
    *   Negative values show negative linearity
    *   If there is a non linear relation, covariance will return 0 since it measures linearity