function [x, varargout] = DET2AnalysisErr2_temp(O, y, sigma_x, sigma_n, q)
% Denoising using DET Approximation
% O - Omega, the dictionary (row normalized)
% y - noisy signal
% sigma_x - signal variance
% sigma_n - noise variance
% stopping criterion by local minima


[p,d] = size(O);
factor = sigma_n^2/sigma_x^2;
qq = log(q/(1-q));
ss = log((1+factor)/factor);


cosupp = false(p,1);
k = 0;
x_i = y./(1+factor);
yty = y'*y;

minterm = (yty)*(factor/(1+factor))/2/sigma_n^2;
err = minterm; % -k* 0.5*ss - k * qq; % this part is zero
err_prev = err + 1;

UO = O';
WO = sum(UO.^2); % This vector should be ones if O is row normalized.
 
U = zeros(d); % Not necessary
r = 1;
if nargout > 2
    errs = zeros(d,1);
    errs(r) = err;
    iter = zeros(d+1,1);
    iter(r) = 7 + d + d + 4 + p*d;    
    XX = zeros(d,d);
    XX(:,r) = x_i;
    order = zeros(d,1);
    order(r) = 0;
end

while err < err_prev && k < p
    term0 = yty -y'*x_i + ((x_i'*UO).*(y'*UO))./WO;
    term0(cosupp) = +Inf;
    terma = term0' /2/sigma_n^2;
    total_mult = d + 2*(p-k)*d + 2*(p-k) + (p-k);
    
    % choose an atom
    LambdaSize = zeros(p,1);
    for i = 1:p
        if cosupp(i) || LambdaSize(i) ~= 0
            continue
        end
        ui = UO(:,i)/sqrt(WO(i));
        aux = ui'*UO;
        lindep = (WO - aux.^2) < 1e-6;
        LambdaSize(lindep) = sum(lindep);
        total_mult = total_mult + d + d*(p-k) + (p-k);
    end
    term = terma - 0.5*LambdaSize*ss - LambdaSize*qq;
    term(cosupp) = +Inf;
    [minterm,jmin] = min(term);
    total_mult = total_mult + p-k + 1;

    urd = UO(:,jmin)/sqrt(WO(jmin));
    
    WO = WO - (urd'*UO).^2;
    cosupp = WO < 1e-6;
    total_mult = total_mult + d + (p-k)*d + p-k;

    UO = UO - urd*(urd'*UO);    
    U(:,r) = urd;
    x_i_prev = x_i;
    x_i = x_i - urd*(urd'*x_i);
    total_mult = total_mult + 2*(p-k)*d + 2*d;

    k = sum(cosupp);
        
    err_prev = err;
    err = minterm;
    
    r = r + 1;
    
    if nargout > 2
        XX(:,r) = x_i;
        iter(r) = total_mult;
        errs(r) = minterm-terma(jmin);
        order(r) = jmin;
    end
end

x = x_i;

if k > 0
    % do one step back if we passed the local minima
    if err > err_prev
        x = x_i_prev;
        if nargout > 2
            XX(:,r) = x;
        end
    end
end

% r = max(r,1);

if nargout > 1
    varargout(1) = {iter(1:r)};
end

if nargout > 2
    XX = XX(:,1:r);
    varargout(2) = {XX};
    varargout(3) = {errs(1:r)};
    varargout(4) = {order(1:r)};
end

end
