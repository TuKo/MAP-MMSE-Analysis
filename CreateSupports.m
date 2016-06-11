function [Omega, k] = CreateSupports(m)

fn = ['CreateSupports' num2str(m) '.mat'];
if exist(fn,'file')
    load(fn);
    return;
end

k = 2^m - 1;

Omega = char(zeros(k+1,m));

parfor i = 0:k
    Omega(i+1,:) = dec2base(i, 2, m);
end

k = k + 1;

save(fn);

end