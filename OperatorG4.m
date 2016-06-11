function [G] = OperatorG4(d1,d2)
% OperatorLandLc -- Constructs the dictionary for difference of gradients.
%
% Input:
%        d1,d2 - size of the image.
%
% Output:
%        G - a d1+d2 by ?? sparse matrix that computes:

% COMMENT: DOES THE LIST OF FIXED PIXELS (BOUNDARY) SHOULD BE TAKEN INTO
% ACCOUNT FOR THE OPERATOR G?

% How to create the general sparse laplacian operator 
m = d2;
mm = m -1;
I = speye(d1);
E = sparse([1:mm,1:mm], [1:mm, 2:mm+1],[ones(1,mm), -ones(1,mm)],mm,mm+1);
Dx = kron(I,E);

m = d1;
mm = m -1;
I = speye(d2);
E = sparse([1:mm,1:mm], [1:mm, 2:mm+1],[ones(1,mm), -ones(1,mm)],mm,mm+1);
Dy = kron(E,I);

m = d2;
mm = m -1;
I = speye(d1);
E = sparse([1:mm,1:mm], [1:mm, 2:mm+1],[ones(1,mm), ones(1,mm)],mm,mm+1);
Dxp = kron(I,E);

m = d1;
mm = m -1;
I = speye(d2);
E = sparse([1:mm,1:mm], [1:mm, 2:mm+1],[ones(1,mm), ones(1,mm)],mm,mm+1);
Dyp = kron(E,I);


G = [Dx; Dy; Dxp; Dyp];

% Normalize the rows of L
G = G * spdiags(1./sqrt(sum(G.^2,2)),0,d1*d2); % ,d1*(d2-1)+d2*(d1-1)

end
