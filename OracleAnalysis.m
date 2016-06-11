function x = OracleAnalysis(O, y, sigma_x, sigma_n, cosupp)
% Denoising using Oracle
% O - Omega, the dictionary (row normalized)
% y - noisy signal
% sigma_x - signal variance
% sigma_n - noise variance
% cosupp - the true cosupport.

d = size(O,2);
In = eye(d);
Os = O(cosupp,:);
OOs = orth(Os');
Qo = In - OOs*OOs';
x = (sigma_n^2/sigma_x^2*In + Qo)\(Qo*y);

end
