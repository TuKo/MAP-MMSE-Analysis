function [term_lambda, support_used] = PrecomputeMMAPTerms(O, q, supps, restore)

[total_SS] = size(supps,1);
[p,d] = size(O);

fn = ['PrecomputeRMAPTerms' num2str(p) '-' restore '.mat'];
if exist(fn,'file')
    load(fn);
    return;
end

In = eye(d);

% Precompute values for MMAPExhaustive
support_used = zeros(total_SS,1);
term_lambda = zeros(total_SS,1);
supp_size = zeros(total_SS,1);
pow2 = 2 .^(p-1:-1:0)';

parfor i = 1:total_SS
    OO = O;
    cosupp = supps(i,:) ~= '0';
    supp_size(i) = sum(cosupp);
    Os = OO(cosupp,:);
    OOs = orth(Os');
    Q = In - OOs*OOs';
    
    % find the support that contains all the linear dependencies that
    % include the current support
    big_support = sum((Q*OO').^2) < 1e-9;
    
    support_used(i) = (big_support * pow2)+1;
end

qq = (q./(1-q));
parfor i = 1:total_SS
    j = support_used(i);
    if (i ~= j) 
        continue;
    end
    term_lambda(i) = sum(qq.^supp_size(support_used == j));
end

% serial code
% for i = 1:total_SS
%     j = support_used(i);
%     term_lambda(j) = term_lambda(j) + qq.^supp_size(i);
% end

save(fn);

end