function [x_mmse,x_map,x_mmap,perf_mmse,perf_map,perf_rmap] = MMSEandMAPandMMAPAnalysisExhaustiveWithPerf(O, y, sigma_x, sigma_n, q, supps, support_used, term_lambda)
% Denoising using Exhaustive MMSE and MAP and Modified MAP
% O - Omega, the dictionary (row normalized)
% y - noisy signal
% sigma_x - signal variance
% sigma_n - noise variance
% q - probability to be in the cosupport of each atom (0<=q<=1)
% supps - the list of possible cosupports.
% support_used - a vector with the representant cosupport of each cosupport
% term_lambda - precomputed  sum(log(q/(1-q))) terms for the group representant

[total_SS] = size(supps,1);
[p,d] = size(O);

In = eye(d);
factor = sigma_n^2/sigma_x^2;
qq = log(q/(1-q));
ss = log((1+factor)/factor);

term = zeros(total_SS,1);
XX = zeros(d,total_SS);

min_lambda = [];
min_term = +Inf;

term2 = zeros(total_SS,1);

r = zeros(total_SS,1);

for i = 1:total_SS
    lambda = supps(i,:) ~= '0';
    l = sum(lambda);
    Os = O(lambda,:);
    OOs = orth(Os');
    Q = In - OOs*OOs';
    C = factor*In+Q;    
    XX(:,i) = (C\(Q*y));
    r(i) = (d-size(OOs,2))*sigma_n^2*sigma_x^2/(sigma_x^2+sigma_n^2);
%     term(i) = -(y'*Q*XX(:,i))/2/sigma_n^2;
    term(i) = (y'*(C\y))/2/sigma_x^2;
    term(i) = term(i) + 0.5*log(det(C));
%     term(i) = term(i) + 0.5*(p-l)*ss;
    term2(i) = term(i);
    if (l>0)
        term(i) = term(i) - l * qq;
    end
    if (term(i) < min_term)
        min_term = term(i);
        min_lambda = lambda;
    end
end

lambda_emap = min_lambda;
Os = O(lambda_emap,:);
OOs = orth(Os');
Qemap = In - OOs*OOs';
x_map = (sigma_n^2/sigma_x^2*In + Qemap)\(Qemap*y);

uniq_support_used = unique(support_used);
newterm = +Inf(size(term2));
newterm(uniq_support_used) = term2(uniq_support_used)-log(term_lambda(uniq_support_used));
[~,min_lambda] = min(newterm);
lambda_emmap = supps(min_lambda,:) ~='0';
Os = O(lambda_emmap,:);
OOs = orth(Os');
Qemmap = In - OOs*OOs';
x_mmap = (sigma_n^2/sigma_x^2*In + Qemmap)\(Qemmap*y);

% if sum(term <0)>0
%     warning('ooh');
% end
term = exp(-term + min_term);
sumterm = sum(term);
termnorm = term /sumterm;
x_mmse = XX*termnorm;

XXperf = XX - repmat(x_mmse, [1, total_SS]);
perf_mmse = termnorm' * (sum(XXperf.^2)' + r);
perf_map  = perf_mmse + sum((x_mmse-x_map).^2);
perf_rmap = perf_mmse + sum((x_mmse-x_mmap).^2);

end
