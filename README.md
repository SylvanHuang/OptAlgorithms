<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Opt-Algorithms

The OptAlgorithms is an optimization toolbox, catering for various optimization problems in different fields. The core part is the classdef, OptAlgorithms.m, which is written with the high-level programming language, MATLAB. It is commenced as one part of author's PhD project at the Institute of Biotechnology of Forschungszentrum Jülich. The author later feel that it will be fantastic to share the advanced methods to certain groups, who are also struggling in the dark of the inverse problems.
    
    ## Algorithms
        * differential evolution (DE);
        * particle swarm optimization (PSO);
        * dayled rejection Markov chain algorihtm (DRAM);
        * Metropolis adjusted differential evolution (MADE);
        * Metropolis adjusted Langevin algorithm, defined on the Riemamn manifold (PRML).

# Introduction

The problems, in which the inputs, outputs, and validated model are given, is considered "inverse" to the forward problem that relates the date (results) that we have observed to the model parameters (causes). However, the inverse problems are ill-posed (due to the definition of Jacques Hadamard), especially in the non-linear cases. The regularization techniques (e.g., truncated SVD, Tikhonov,  iterative method) are typically used to tackle the ill-posed issues in the linear problems.

Our toolbox is especially tailored for the non-linear problems by employing the heuristics (e.g., DE, PSO) and Bayesian inference. 

In the heuristic field, DE and PSO are regarded as the most powerful ones, as they can converge to the acceptable points in the desirable computational time. However, either the inputs and outputs of the system is error-prone. Thus it is not convincing to merely provide the single parameter vector, rather than the distributions of the optimized parameters. This is the reason why the Bayesian inference play around in the inverse problems.

In Bayesian inference field, Markov chain Monte Carlo (MCMC) is used to sample the posterior distribution of parameters, which is (commonly) multi-modal. The frequently adopted algorithm, Metropolis-Hasttings, is certainly included in the package, and it is also extended by the strategy of delayed rejection to boost the acceptance ratio, leading to the delayed rejection adjusted Metropolis algorithm (DRAM). However the DRAM has a weakness that it might trap into the single modal without jumping to another modal. The Metropolis adjusted differential evolution (MADE) is subsequently developed from the differential evolution Markov chain (DEMC) to overcome above drawback, by replacing the proposal distribution of the DRAM with the differential evolution (DE) kernel to enhance the global roaming. The last algorithm, PRML, is quite state-of-art since it introduces the Riemann geometry into the Bayesian field. The use of the Riemann manifold allow us to define a measure of the distance between parameter distributions in terms of the change in target distribution, rather than changes in the value of parameters in Euclidean space. Thus the PRML algorithm takes small steps in directions of high sensitivity of the target distribution and bigger steps in directions of low sensitivity of the target distribution.  


# Feature list

    * Flexibility. Adaption for various problems;
    * Range. Ranging from deterministic, heuristic algorithms to Bayesian inference;
    * Easy-to-use. Acting as a blackbox, only need to offer searching domain and parameter number; 

# Dependency and Platforms

    * Matlab(R2010b or higher);

# Demonstration 

The Ackley function is used to demonstrate, which is widely used for testing optimization algorithms, as shown below. It is characterized by a nearly flat outer region, and a large hole at the centre. The function poses a risk for optimization algorithms, particularly hill-climbing (deterministic) algorithms, to be trapped in one of its many local minima.

![](https://github.com/KimHe/OptAlgorithms/blob/master/doc/ackley_equ.png)

![](https://github.com/KimHe/OptAlgorithms/blob/master/doc/ackley.png)
*Ackley Function*

The searching domain is [-32.678, 32.678] for i = 1,2,...,d;
The global optimum is f([0,0,...,0]) = 0;

# Further Development 

The OptAlgorithms is actively developed. Hence, breaking changes and extensive restructuring may occur in any commit and release. For non-developers it is recommended to upgrade from release to release instead of always working with the most recent commit.

If you find any bug, please do not be hesitate to email the author by kingdomhql@gmail.com.
