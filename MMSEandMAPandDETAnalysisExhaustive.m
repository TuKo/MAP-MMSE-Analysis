function [x_mmse,x_map,x_det,x_mmap,varargout] = MMSEandMAPandDETAnalysisExhaustive(O, y, sigma_x, sigma_n, q, supps, support_used, term_lambda)
% Denoising using Exhaustive MMSE and MAP and RMAP and wrong AMP (alias DET)
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
min_lambda2 = [];
min_term2 = +Inf;

term2 = zeros(total_SS,1);
term3 = zeros(total_SS,1);

for i = 1:total_SS
    lambda = supps(i,:) ~= '0';
    l = sum(lambda);
    Os = O(lambda,:);
    OOs = orth(Os');
    Q = In - OOs*OOs';
    C = factor*In+Q;    
    XX(:,i) = (C\(Q*y));

    term(i) = (y'*(C\y))/2/sigma_x^2;
    term2(i) = term(i);
    
    term(i) = term(i) + 0.5*log(det(C));
    term2(i) = term2(i) + 0.5*(p-l)*ss;
    term3(i) = term(i);
    if (l>0)
        term(i) = term(i) - l * qq;
        term2(i) = term2(i) - l * qq;
    end
    if (term(i) < min_term)
        min_term = term(i);
        min_lambda = lambda;
        min_lambda_i = i;
    end
    if (term2(i) < min_term2)
        min_term2 = term2(i);
        min_lambda2 = lambda;
    end
end

lambda_emap = min_lambda;
Os = O(lambda_emap,:);
OOs = orth(Os');
Qemap = In - OOs*OOs';
x_map = (sigma_n^2/sigma_x^2*In + Qemap)\(Qemap*y);
if nargout > 4
    varargout(1) = {lambda_emap};
end

uniq_support_used = unique(support_used);
newterm = +Inf(size(term3));
newterm(uniq_support_used) = term3(uniq_support_used)-log(term_lambda(uniq_support_used));
[~,min_lambda] = min(newterm);
lambda_emmap = supps(min_lambda,:) ~='0';
Os = O(lambda_emmap,:);
OOs = orth(Os');
Qemmap = In - OOs*OOs';
x_mmap = (sigma_n^2/sigma_x^2*In + Qemmap)\(Qemmap*y);
if nargout > 5
    varargout(2) = {lambda_emmap};
end

lambda_det = min_lambda2;
Os = O(lambda_det,:);
OOs = orth(Os');
Qdet = In - OOs*OOs';
x_det = (sigma_n^2/sigma_x^2*In + Qdet)\(Qdet*y);
if nargout > 6
    varargout(3) = {lambda_det};
end

% if sum(term <0)>0
%     warning('ooh');
% end
term = exp(-term + min_term);
sumterm = sum(term);
termnorm = term /sumterm;
x_mmse = XX*termnorm;

end
