function [G] = OperatorG2(d1,d2)
% OperatorL -- Constructs the dictionary for difference of gradients.
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


mm = d1*d2;
Dxy = eye(d1*d2);
Dxy(sub2ind(size(Dxy), 1:mm, mod((1:mm)+d2,mm)+1)) = -1;
%remove the cyclic part
% Dxy = Dxy(1:end-d2,:);
stay = true(size(Dxy,1),1);
stay(d1*(1:(d2-1))) =false;
stay((end-d2):end) = false;
Dxy = Dxy(stay,:);

mm = d1*d2;
Dyx = eye(d1*d2);
Dyx(sub2ind(size(Dyx), 1:mm, mod((1:mm)-d2,mm)+1)) = -1;
%remove the cyclic part
stay = true(size(Dyx,1),1);
stay(d1*(2:(d2))) =false;
stay(1:d2) = false;
Dyx = Dyx(stay,:);


G = [Dx; Dy; Dxy; Dyx];
% G = [G; sparse( ones(d,1), 1:d, ones(d,1))];

% Normalize the rows of L
G = G * spdiags(1./sqrt(sum(G.^2,2)),0,d1*d2); % ,d1*(d2-1)+d2*(d1-1)

end
