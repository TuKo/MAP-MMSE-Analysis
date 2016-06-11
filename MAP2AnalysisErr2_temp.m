function [x, varargout] = MAP2AnalysisErr2_temp(O, y, sigma_x, sigma_n, q)
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
err = minterm + 0.5*p*ss; % -k* 0.5*ss - k * qq; % this part is zero
err_prev = err + 1;

UO = O';
WO = sum(UO.^2); % This vector should be ones if O is row normalized.

U = zeros(d);
t = 1;
if nargout > 2
    iter = zeros(d+1,1);
    iter(t) = 7 + d + d + 6 + p*d;
    XX = zeros(d,d);
    XX(:,t) = x_i;
end


while err < err_prev && k < p
    
    terma0 = yty -y'*x_i + ((x_i'*UO).*(y'*UO))./WO;
    terma0(cosupp) = +Inf;
    terma0 = terma0' /2/sigma_n^2;
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
    terma = terma0 - LambdaSize*qq + 0.5*(p-t)*ss;
    terma(cosupp) = +Inf;
    [minterm,jmin] = min(terma);
    total_mult = total_mult + p-k + 2;
    
    urd = UO(:,jmin)/sqrt(WO(jmin));
    
    WO = WO - (urd'*UO).^2;
    cosupp = WO < 1e-6;
    total_mult = total_mult + d + (p-k)*d + p-k;    

    UO = UO - urd*(urd'*UO);    
    U(:,t) = urd;
    x_i_prev = x_i;
    x_i = x_i - urd*(urd'*x_i);
    total_mult = total_mult + 2*(p-k)*d + 2*d;

    k = sum(cosupp);
    
    err_prev = err;
    err = minterm;
    
    t = t + 1;
    
    if nargout > 2
        XX(:,t) = x_i;
        iter(t) = total_mult;
    end
end

x = x_i;

if k > 0
    % do one step back if we passed the local minima
    if err > err_prev
        x = x_i_prev;
        if nargout > 2
            XX(:,t) = x;
        end
    end
end

% t = max(t,1);

if nargout > 1
    varargout(1) = {iter(1:t)};
end

if nargout > 2
    XX = XX(:,1:t);
    varargout(2) = {XX};
end

end
