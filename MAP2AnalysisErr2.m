function [x, varargout] = MAP2AnalysisErr2(O, y, sigma_x, sigma_n, q)
% Denoising using MAP Approximation
% O - Omega, the dictionary (row normalized)
% y - noisy signal
% sigma_x - signal variance
% sigma_n - noise variance
% stopping criterion by local minima

[p,d] = size(O);
I = eye(d);
factor = sigma_n^2/sigma_x^2;
qq = log(q/(1-q));
ss = log((1+factor)/factor);

Q = I;
C = Q+factor*I;
x = C\(Q*y);

cosupp = false(p,1);
cosupp_wodependencies = false(p,1);
k = 0;
iter = 0;

minterm = (y'*(C\y))/2/sigma_x^2;
err = minterm + 0.5*p*ss; % - k * qq; % this part is zero
err_prev = err + 1;
if nargout > 2
    XX = zeros(d,d);
    XX(:,iter+1) = x;
end

while err < err_prev && k < p
    % choose an atom
    term = zeros(p,1);
    terma0 = zeros(p,1);
    r = sum(cosupp_wodependencies);
    for i = 1:p
        if cosupp(i)
            term(i) = +Inf;
            continue
        end
        
        cosupp_new = cosupp_wodependencies;
        cosupp_new(i) = true;
        OO = O(cosupp_new,:);
        UU = orth(OO');
        Q_new = I - UU*UU';
        C_new = Q_new+factor*I;
        terma0(i) = (y'*(C_new\y))/2/sigma_x^2;
        LambdaSize = sum(sum((Q_new*O').^2) < 1e-6);
        term(i) = terma0(i) + 0.5*(p-(r+1))*ss;
        term(i) = term(i) - LambdaSize*qq;        
    end
    
    [minterm,jmin] = min(term);
    cosupp(jmin) = true;
    cosupp_wodependencies(jmin) = true;
    % need to reduce the number of zeros.

    % compute the new x for the stopping criterion
    OO = O(cosupp,:);
    UU = orth(OO');
    Q = I - UU*UU';   

    cosupp = abs(sum((Q*O').^2)) < 1e-6;
        
    k = sum(cosupp);
    err_prev = err;
    err = minterm;
    
    iter = iter + 1;

    if nargout > 2
        C = Q+factor*I;
        x = C\(Q*y);    
        XX(:,iter+1) = x;
    end    
end

if k > 0
    % do one step back if we passed the local minima
    if err > err_prev
        cosupp_wodependencies(jmin) = false;
    else
        iter = iter + 1;
    end
    OO = O(cosupp_wodependencies,:);
    UU = orth(OO');
    Q = I - UU*UU';
    C = Q+factor*I;
    x = C\(Q*y);
end

if nargout > 1
    varargout(1) = {iter};
end

if nargout > 2
    XX = XX(:,1:iter);
    varargout(2) = {XX};
end

end
