function x = DET2AnalysisErr2lambda(O, y, sigma_x, sigma_n, lambda0)
% Denoising using MAP Approximation
% O - Omega, the dictionary (row normalized)
% y - noisy signal
% sigma_x - signal variance
% sigma_n - noise variance
% stopping criterion by local minima
% With MGS orthogonalization

[p,d] = size(O);
I = eye(d);
factor = sigma_n^2/sigma_x^2;

Q = I;
C = Q+factor*I;
x = C\(Q*y);

cosupp = false(p,1);
cosupp_wodependencies = false(p,1);
k = 0;

minterm = (y'*(C\y))/2/sigma_x^2;
err = minterm; % -k*lambda0; % this part is zero
err_prev = err + 1;
r = 1;

while err < err_prev && k < p
    % choose an atom
    term = zeros(p,1);
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
        term(i) = (y'*(C_new\y))/2/sigma_x^2;
        LambdaSize = sum(sum((Q_new*O').^2) < 1e-6);
        term(i) = term(i) - lambda0*LambdaSize;
    end
    
    [minterm,jmin] = min(term);
    
    cosupp(jmin) = true;
    cosupp_wodependencies(jmin) = true;
    % need to reduce the number of zeros.

    % compute the new x for the stopping criterion
    OO = O(cosupp,:);
    UU = orth(OO');
    Q = I - UU*UU';
    
    cosupp = sum((Q*O').^2) < 1e-6;        
    r = r + 1;
    k = sum(cosupp);
    err_prev = err;
    err = minterm;
end

if k > 0
    % do one step back if we passed the local minima
    if err > err_prev
        cosupp_wodependencies(jmin) = false;
    end
    OO = O(cosupp_wodependencies,:);
    UU = orth(OO');
    Q = I - UU*UU';
    C = Q+factor*I;
    x = C\(Q*y);
end

end
