function [x, cosupp, varargout] = batchobgErr2(O, y, lambda_0, lambda_2)
%FASTBG    Computes an approximate solution to the Analysis model problem:
%
%   X = arg min_z ||Y - z||_2^2 + LAMBDA_2*||z||_2^2
%         s.t.     ||O*z||_0 <= P-L
% 
%   where P is the number of rows in O.
%
%   The solution is computed using the Backward Greedy (BG). The code is
%   optimized for fast computations. The algorithm time complexity is the
%   same as matrix by vector multiplication.
%
%   [X, COSUPP] = FASTBG(O, Y, L, LAMBDA_2) returns the approximate 
%   solution X of the problem described above. The co-support is returned
%   as well in COSUPP.
%   The parameters are the following:
%       O: the analysis dictionary Omega (row normalized).
%       Y: the noisy signal.
%       L: number of expected atoms in the co-support (stopping criterion).
%       LAMBDA_2: L2 regularizer. 
%
%   The actual effect of LAMBDA_2 is of shrinkage due to the noise effects.
%   Use LAMBDA_2 equal to zero to avoid the shrinkage effect.

% File version: 1.02

% Copyright 2011, Javier Turek, Dept. of Computer Science, Technion

[p,d] = size(O);

UO = O';
W = sum(UO.^2); % This vector should be ones if O is row normalized.

x_i = y./(1+lambda_2);
cosupp = false(p,1);
r = 1;
U = zeros(d,d);
k = 0;

err = lambda_2 * (x_i'*x_i) + lambda_0 * (p-sum(cosupp)) + (y-x_i)'*(y-x_i);
err_prev = err + 1;
if nargout > 3
    iter = zeros(p+1,1);
    iter(r) = 3 + d + d + 1 + d + 1 +p*d;
    XX = zeros(d,d);
    XX(:,r) = x_i;
end

while err < err_prev && r < p
    % Select the most orthonormal element
    proj = (1+lambda_2)*((O*x_i).^2)./W';
    proj(cosupp) = Inf; % Remove the elements that are already in the cosupp
    
    total_mult = (p-k)*d + p-k + p-k + p-k;
    
    [~,j] = min(proj);

    % If the element choosen is linearly dependent with others in the
    % co-support, then only add it.
%     if (abs(W(j)) < 1e-12)
%         ui = zeros(d,1);
%     else
    ui = (O(j,:)' - U*(U'*O(j,:)'))/sqrt(W(j));
    total_mult = total_mult + 2* r *d + d;
%     end
    U(:,r) = ui;


    % Gram-Schmidt step
    aux = (O*ui)';
    W = W - aux.^2;
    total_mult = total_mult + (p-k)*d + (p-k);
    
    % Update the co-support
    cosupp = W < 1e-6;
    k = sum(cosupp);
    
    % Compute the new estimate
    x_iPrev = x_i;
    x_i = x_i - ui*(ui'*x_i);
    total_mult = total_mult + 2*d;
    
    
    r = r + 1;
    err_prev = err;
    err = lambda_2 * (x_i'*x_i) + lambda_0 * (p-k) + (y-x_i)'*(y-x_i);
    
    total_mult = total_mult + d + 2 + d;
    
%     fprintf('r=%3d, err=%f\n',r,err);
    
    if nargout > 3
        iter(r) = total_mult;
        XX(:,r) = x_i;
    end
end

if r > 0 && err > err_prev 
    x_i = x_iPrev;
    XX(:,r) = x_i;
end
x = x_i;

if nargout > 2
    varargout(1) = {iter(1:r)};
end

if nargout > 3
    XX = XX(:,1:r);
    varargout(2) = {XX};
end

end