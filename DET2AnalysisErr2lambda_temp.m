function x = DET2AnalysisErr2lambda_temp(O, y, sigma_x, sigma_n, lambda0)
% Denoising using MAP Approximation
% O - Omega, the dictionary (row normalized)
% y - noisy signal
% sigma_x - signal variance
% sigma_n - noise variance
% stopping criterion by local minima but using the parameter lambda0 
% With MGS orthogonalization

[p,d] = size(O);
factor = sigma_n^2/sigma_x^2;

cosupp = false(p,1);
k = 0;
x_i = y./(1+factor);
yty = y'*y;

minterm = (yty)*(factor/(1+factor))/2/sigma_n^2;
err = minterm; % -k* 0.5*ss - k * qq; % this part is zero
err_prev = err + 1;

U = zeros(d); % Not necessary
r = 1;

UO = O';
WO = sum(UO.^2); % This vector should be ones if O is row normalized.

while err < err_prev && k < p
    term0 = yty -y'*x_i + ((x_i'*UO).*(y'*UO))./WO;
    term0(cosupp) = +Inf;
    terma = term0' /2/sigma_n^2;

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
    end
    terma = terma - lambda0*LambdaSize;
    terma(cosupp) = +Inf;
    [minterm,jmin] = min(terma);

    % compute the new x for the stopping criterion
    urd = UO(:,jmin)/sqrt(WO(jmin));
    
    WO = WO - (urd'*UO).^2;
    cosupp = WO < 1e-6;

    UO = UO - urd*(urd'*UO);    
    U(:,r) = urd;
    x_i_prev = x_i;
    x_i = x_i - urd*(urd'*x_i);

    k = sum(cosupp);
        
    err_prev = err;
    err = minterm;
    
    r = r + 1;
end

x = x_i;

if k > 0
    % do one step back if we passed the local minima
    if err > err_prev
        x = x_i_prev;
    end
end


end
