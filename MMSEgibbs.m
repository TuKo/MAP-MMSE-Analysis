function [x_mmse, XX, varargout] = MMSEgibbs(O, y, sigma_x, sigma_n, q, iter)

% Problem size
[p,d] = size(O);

Id = eye(d);
cs = sigma_n^2/sigma_x^2;

randstream = rand(iter,1);

% Draw a temporary support using q
% L = rand(p,1) < q; %Random initialization
L = false(p,1); %Zero initialization
z = 1;
XX = zeros(d,iter);

Q = Id;
C = Id*cs+Id;
XX(:,z) = C\(Q*y);
sumX = XX(:,z);
last_sumX = XX(:,z)+1; % make sure sumX/z ~= last_sumX

while z<iter && sum((sumX/z-last_sumX).^2) > 1e-10
    last_sumX = sumX/z;
    
    i = mod(z-1,p)+1;
    Lin = L;
    Lin(i) = true;    
    Uin = orth(O(Lin,:)');
    Qin = Id - Uin*Uin';
    Cin = cs*Id + Qin;
    
    Lout = L;
    Lout(i) = false;
    Uout = orth(O(Lout,:)');
    Qout = Id - Uout*Uout';
    Cout = cs*Id + Qout;

    prob_in = q/sqrt(det(Cin))*exp(-y'*(Cin\y)/2/sigma_x^2);
    prob_out = (1-q)/sqrt(det(Cout))*exp(-y'*(Cout\y)/2/sigma_x^2);
    prob = prob_in + prob_out;
    L(i) = randstream(z) < (prob_in/prob);
    
    z = z+1;
    if L(i) == true
        XX(:,z) = Cin\(Qin*y);
    else
        XX(:,z) = Cout\(Qout*y);
    end
    sumX = sumX + XX(:,z);
end

% z = z - 1;
% if z > 0
    % MMSE: mean over all the solutions
    XX = XX(:,1:z);
%     x_mmse = mean(XX,2);
    x_mmse = sumX/z;

    XX = cumsum(XX,2);
    XX = XX ./ repmat(1:z, [d,1]);
% else
%     x_mmse = zeros(d,1);
%     XX = [];
% end

if nargout>2
    varargout(1) = {z};
end

end