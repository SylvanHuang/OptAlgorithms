function y = objectiveFunc(xx, a, b, c)
%==============================================================================
% ACKLEY FUNCTION
%
% Authors: Sonja Surjanovic, Simon Fraser University
%          Derek Bingham, Simon Fraser University
% Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
%
% Copyright 2013. Derek Bingham, Simon Fraser University.
%
% For function details and reference information, see:
% http://www.sfu.ca/~ssurjano/
%
% Parameters:
%       - xx = [x1, x2, ..., xd]
%       - a = constant (optional), with default value 20
%       - b = constant (optional), with default value 0.2
%       - c = constant (optional), with default value 2*pi
%
% Return:
%       y = objective function value
%==============================================================================


    d = length(xx);

    if nargin < 2
        a = 20;
    end
    if nargin < 3
        b = 0.2;
    end
    if nargin < 4
        c = 2 * pi;
    end

    sum1 = 0; sum2 = 0;

    for ii = 1:d
        xi = xx(ii);
        sum1 = sum1 + xi^2;
        sum2 = sum2 + cos(c*xi);
    end

    term1 = -a * exp(-b*sqrt(sum1/d));
    term2 = -exp(sum2/d);

    y = term1 + term2 + a + exp(1);

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
