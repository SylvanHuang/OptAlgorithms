function run()
% =============================================================================
% Optimized variables
%       theta = {x_1, x_2, ...}
%
% There are five types of algorithms that are integrated into this classdef, ranging
% from deterministic, heuristic algorithms to Bayesian Inference:
%       - Particle Swarm Optimizatio (PSO)
%       - Differential Evolution (DE)
%       - Markov chain Monte Carlo (MCMC) (Jacobian matrix might be needed)
%       - Metropolis Adjusted Differential Evolution (MADE)
%       - Metropolis Adjusted Langevin Algorithm (MALA)
%           defined on the Riemann geometry, and combined with parallel tempering
%           Jacobian matrix must be needed
% =============================================================================


    % There are four optimization algorithms availabe in this programme
    optimization_method = struct('Particle_Swarm_Optimization',[], 'Differential_Evolution',[],...
       'Metropolis_Adjusted_Differential_Evolution',[], 'Riemann_Manifold_Metropolis_Adjusted_Langevin',[],...
       'Markov_Chain_Monte_Carlo',[], 'Deterministic_algorithm_fmincon',[]);

    % The set of the parameters which are optimized
    params = struct('x_1',[], 'x_2',[]);

    % The initial boundary of parameters: In the format of [x^1_min x^1_max; ...]
    opt.paramBound = [-32.768 32.768; -32.768 32.768];

    % Check the consistence of the initial boundary condition and the parameter amount
    OptAlgorithms.checkOptDimension(opt, length(fieldnames(params)));

    % Select one method and make it true (correspondingly the rest methods false)
    optimization_method.Differential_Evolution = false;
    optimization_method.Particle_Swarm_Optimization = false;
    optimization_method.Deterministic_algorithm_fmincon = false;
    optimization_method.Markov_Chain_Monte_Carlo = false;
    optimization_method.Metropolis_Adjusted_Differential_Evolution = true;
    optimization_method.Parallel_Riemann_Metropolis_Adjusted_Langevin = false;


    if isfield(optimization_method, 'Particle_Swarm_Optimization') ...
            && optimization_method.Particle_Swarm_Optimization

        OptAlgorithms.Particle_Swarm_Optimization(opt, params);

    elseif isfield(optimization_method, 'Differential_Evolution') ...
            && optimization_method.Differential_Evolution

        OptAlgorithms.Differential_Evolution(opt, params);

    elseif isfield(optimization_method, 'Markov_Chain_Monte_Carlo') ...
            && optimization_method.Markov_Chain_Monte_Carlo

        OptAlgorithms.Markov_Chain_Monte_Carlo(opt, params);

    elseif isfield(optimization_method, 'Metropolis_Adjusted_Differential_Evolution') ...
            && optimization_method.Metropolis_Adjusted_Differential_Evolution

        OptAlgorithms.Metropolis_Adjusted_Differential_Evolution(opt, params);

    elseif isfield(optimization_method, 'Parallel_Riemann_Metropolis_Adjusted_Langevin')

        OptAlgorithms.Parallel_Riemann_Metropolis_Adjusted_Langevin(opt, params);

    elseif isfield(optimization_method, 'Deterministic_algorithm_fmincon') ...
            && optimization_method.Deterministic_algorithm_fmincon

        % The guessing point for the deterministic method 
        initParams = [10 10];

        % Check the consistence of the initial boundary condition and the parameter amount
        OptAlgorithms.checkOptDimension(opt, length(initParams));

        loBound = opt.paramBound(:,1);
        upBound = opt.paramBound(:,2);

        options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'iter',...
            'TolX',1e-6,'TolCon',1e-6,'TolFun',1e-6,'MaxIter',500);

        try
            [xValue, yValue, exitflag, output, ~, grad] = fmincon( @objectiveFunc, ...
                initParams, [],[],[],[], loBound, upBound, [], options);
        catch exception
            disp('Errors in the MATLAB build-in optimizer: fmincon. \n Please check your input parameters and run again. \n');
            disp('The message from fmincon: %s \n', exception.message);
        end

        fprintf('----------------  Minimum: %10.3g  ---------------- \n', yValue);
        fprintf('%10.3g | ', xValue);
        fprintf('\n------------------------------------------------------ \n');

    else

        error('The method you selected is not provided in this programme \n');

    end

end
% =============================================================================
%              The MATLAB library for optimization case studies
% 
%      Copyright © 2015-2016: Qiaole He
% 
%      Forschungszentrum Juelich GmbH, IBG-1, Juelich, Germany.
% 
%  All rights reserved. This program and the accompanying materials
%  are made available under the terms of the GNU Public License v3.0 (or, at
%  your option, any later version) which accompanies this distribution, and
%  is available at http://www.gnu.org/licenses/gpl.html
% =============================================================================
