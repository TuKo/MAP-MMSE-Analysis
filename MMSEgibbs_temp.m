function [x_mmse, XX, varargout] = MMSEgibbs_temp(O, y, sigma_x, sigma_n, q, max_iter, varargin)

% Problem size
[p,d] = size(O);

cs = sigma_n^2/sigma_x^2;
detConst = (1+cs)^d;
detCoeff = cs/(1+cs);

if (nargin <= 6)
    randstream = rand(max_iter,1);
else
    randstream = varargin{1};
end

% Draw a temporary support using q
% L = rand(p,1) < q; %Random initialization
L = false(p,1); %Zero initialization
z = 1;
XX = zeros(d,max_iter);

order = zeros(p,1);
U = zeros(d);
next_atom = 1;

pos = zeros(d,1);
detPrev = detConst;
errPrev = -(y'*y)/2/sigma_x^2/(1+cs);
x_i = y./(1+cs);
Uprev = cell(p,1);
errPrevList = zeros(p,1);
detPrevList = zeros(p,1);
XXprev = zeros(d,p);

XX(:,z) = x_i;
sumX = x_i;
last_sumX = x_i+1;  % make sure sumX/z ~= last_sumX

if nargout > 2
    iter = zeros(max_iter,1);
    iter(z) = 5 + d + 3 + d;
end

while z<max_iter %&& sum((sumX/z-last_sumX).^2) > 1e-10
    last_sumX = sumX/z;
    i = mod(z-1,p)+1;
    total_mult = 2*d + d; % includes the condition of the while
    
    if L(i)
        % The element was in the support:
        % The IN part remains the same as before.
        detIn = detPrev;
        errIn = errPrev;

        % The OUT part depends if the atom "i" was added as linearly 
        % dependent with previous added atoms that were inside the
        % cosupport before it was added.
        if order(i) == 0 
            % If it was dependent of previously added atoms, then removing
            % it won't affect the determinant and the error.
            detOut = detPrev;
            errOut = errPrev;
        else
            % If it was independent when it was added, then we need to
            % check that other additions to the cosupport are not linearly
            % dependent with the direction that we may remove.
            % Also, the OUT part needs to take the U matrix at the moment the atom
            % was previously added into the solution and restore the solution
            % without the atom.
        Uprevaux = Uprev;
        orderAux = order;
        errAux = errPrevList;
        detAux = detPrevList;
        
        Uprevaux{i} = [];
        orderAux(i) = 0;
        
        Uaux = Uprev{i};
        errOut = errAux(i);
        detOut = detAux(i);    
        next_aux = order(i); 
        x_aux = XXprev(:,i);
        XXprevaux = XXprev;
        XXprevaux(:,i) = zeros(d,1);
        
        elems = find(pos > pos(i));
        [~,idx] = sort(pos(elems));
        elem_ordered = elems(idx);
        
        for count = 1:numel(elem_ordered)
            j = elem_ordered(count);
            Uprevaux{j} = Uaux;
            errAux(j) = errOut;
            detAux(j) = detOut;
            orderAux(j) = 0;
            XXprevaux(:,j) = x_aux;
            
            % Update U
            wj = O(j,:)';
            uj = wj - Uaux*(Uaux' * wj);
            Wj = uj'*uj;
            isDependent = Wj < 1e-10;
            total_mult = total_mult + 2*(next_aux-1)*d + d;
            
            if ~isDependent
                uj = uj ./ sqrt(Wj);
                x_aux = x_aux - uj*(uj'*x_aux);
                Uaux(:,next_aux) = uj;
                orderAux(j) = next_aux;
                next_aux = next_aux + 1;
                % Update the error and the determinant
                detOut = detOut * detCoeff;
                errOut = errOut - ((uj'*y)^2)/2/sigma_n^2/(1+cs);
                
                total_mult = total_mult + 3*d + 1 + d + 2;
            end
        end

        end
    else
        % The element was not in the support:
        % The OUT part remains the same as before.
        detOut = detPrev;
        errOut = errPrev;

        % The IN part depends if the atom "i" is learly dependent with the
        % other atoms that are inside the cosupport
        wi = O(i,:)';
        ui = wi - U*(U' * wi);
        Wi = ui'*ui;
        isDependent = Wi < 1e-10;
        total_mult = total_mult + 2*(next_atom-1)*d + d;
        if isDependent
            % If it is dependent with what can be found in U then the error
            % and the determinant don't change.
            detIn = detPrev;
            errIn = errPrev;
        else
            % If it is independent, then we need to update the values of
            % the error and the determinant using the recursive formula.
            ui = ui ./ sqrt(Wi);
            detIn = detPrev * detCoeff;
            errIn = errPrev - ((ui'*y)^2)/2/sigma_n^2/(1+cs);
            total_mult = total_mult + d + 1 + d + 2;
        end        
    end

    prob_in = q/sqrt(detIn)*exp(errIn);
    prob_out = (1-q)/sqrt(detOut)*exp(errOut);

    prob = prob_in + prob_out;
    Li = randstream(z) < (prob_in/prob);
    total_mult = total_mult + 5;
    
    if Li
        if ~L(i)
            errPrevList(i) = errPrev;
            detPrevList(i) = detPrev;
            pos(i) = z; % only update pos if it was not in the cosupport.
            Uprev{i} = U; % Add U before adding the new atom
            XXprev(:,i) = x_i;
            
            if ~isDependent
                x_i = x_i - ui*(ui'*x_i);
                U(:,next_atom) = ui;
                order(i) = next_atom;
                next_atom = next_atom + 1;
                total_mult = total_mult + 2*d;
            end
        end
        detPrev = detIn;
        errPrev = errIn;
    else
        detPrev = detOut;
        errPrev = errOut;
        pos(i) = 0;
        if L(i) && order(i) > 0
            XXprev = XXprevaux;
            x_i = x_aux;
            Uprev = Uprevaux;
            errPrevList = errAux;
            detPrevList = detAux;
            U = Uaux;
            order = orderAux;
            next_atom = max(order) + 1;
        end
        order(i) = 0;
    end
    
    z = z + 1;
    if nargout > 2
        iter(z) = total_mult;
    end
    XX(:,z) = x_i;
    L(i) = Li;
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
    iter(z) = iter(z) + 2*d;
    varargout(1) = {iter(1:z)};
end

end