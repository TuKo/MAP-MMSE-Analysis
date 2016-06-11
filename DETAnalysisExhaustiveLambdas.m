function [x_det] = DETAnalysisExhaustiveLambdas(O, y, sigma_x, sigma_n, supps, regularizers)
% Denoising using Exhaustive MMSE and MAP and RMAP and wrong AMP (alias DET)
% O - Omega, the dictionary (row normalized)
% y - noisy signal
% sigma_x - signal variance
% sigma_n - noise variance
% q - probability to be in the cosupport of each atom (0<=q<=1)
% supps - the list of possible cosupports.
% regularizers - vector with regularizer values

[total_SS] = size(supps,1);
[p,d] = size(O);

In = eye(d);
factor = sigma_n^2/sigma_x^2;

term = zeros(1,total_SS);
term2 = zeros(1,total_SS);
XX = zeros(d,total_SS);

for i = 1:total_SS
    lambda = supps(i,:) ~= '0';
    l = sum(lambda);
    Os = O(lambda,:);
    OOs = orth(Os');
    Q = In - OOs*OOs';
    C = factor*In+Q;    
    XX(:,i) = (C\(Q*y));

    term(i) = (y'*(C\y))/2/sigma_x^2;
    term2(i) = (p-l);
end

total_reg = numel(regularizers);
term_per_reg = ones(total_reg,1) * term;
term_per_reg = term_per_reg + repmat(regularizers,[1,total_SS]) .* (ones(total_reg,1)* term2);
[~,idxs] = min(term_per_reg,'',2);
x_det = XX(:,idxs);

end
