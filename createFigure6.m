function createFigure6(O, q, sigma_n, filename, max_iter)

% Experiment setup

% Problem size
[p,d] = size(O);

% Model parameters
sigma_x = sqrt(1);
In = eye(d);

fprintf('Problem size p=%d, d=%d, q=%g\n',p,d,q);

% Draw a noisy signal.
cosupp = rand(p,1) < q;
cosupp_size = sum(cosupp);
if (cosupp_size > 0)
    Os = O(cosupp,:);
    Os = orth(Os'); % Use orthogonalization to avoid zero columns.
    Q = In - Os*Os';
else
    Q = In;
end
fprintf('|Lambda_orig| = %d\n',cosupp_size);
fprintf('r = %d\n',size(Os,2));

x0 = randn(d,1) * sigma_x;
x = Q * x0;
y = x + sigma_n*randn(d,1);

x_oracle = OracleAnalysis(O,y,sigma_x,sigma_n,cosupp);
MSE_oracle = sum((x_oracle-x).^2);


[x_det,iter_det,XX_det] = DET2AnalysisErr2_temp(O, y, sigma_x, sigma_n, q);
MSE_det = sum((x_det-x).^2)
iter_det_l = size(XX_det,2);

[x_map,iter_map,XX_map] = MAP2AnalysisErr2_temp(O, y, sigma_x, sigma_n, q);
MSE_map = sum((x_map-x).^2)
iter_map_l = size(XX_map,2);

lambda_2 = sigma_n^2/sigma_x^2;
lambda_0 = 2*sigma_n^2 * log(q/(1-q)*sqrt((lambda_2+1)/lambda_2));
[x_obg, ~, iter_obg, XX_obg] = batchobgErr2(O, y, lambda_0, lambda_2);
MSE_obg = sum((x_obg-x).^2)
iter_obg_l = size(XX_obg,2);

randstream = rand(max_iter,1);
[x_mmse, XX_mmse,iter_mmse] = MMSEgibbs_temp(O, y, sigma_x, sigma_n, q, max_iter,randstream);
MSE_mmse = sum((x_mmse-x).^2)
iter_mmse_l = size(XX_mmse,2);

save(filename)

% Convergence graph
figure;
semilogx(cumsum(iter_mmse)/(p*d), sum((XX_mmse-repmat(x,[1,iter_mmse_l])).^2)./(d*sigma_n^2),'-k','Linewidth',2,'MarkerSize',8);
hold on;
plot(cumsum(iter_obg)/(p*d), sum((XX_obg-repmat(x,[1,iter_obg_l])).^2)./(d*sigma_n^2),'-r','Linewidth',2,'MarkerSize',8);
plot(cumsum(iter_map)/(p*d), sum((XX_map-repmat(x,[1,iter_map_l])).^2)./(d*sigma_n^2),'-b','Linewidth',2,'MarkerSize',8);
plot(cumsum(iter_det)/(p*d), sum((XX_det-repmat(x,[1,iter_det_l])).^2)./(d*sigma_n^2),'-g','Linewidth',2,'MarkerSize',8);
plot([min(iter_mmse), sum(iter_mmse)]/(p*d),[MSE_oracle, MSE_oracle]/(d*sigma_n^2),'--m');
% 
% plot([1, p],[MSE_det, MSE_det],'-r');
% plot([p, p],[0, MSE_det],'--r');
% plot(p, MSE_det,'xr');
% 
% plot([1, p],[MSE_map, MSE_map],'-b');
% plot([p, p],[0, MSE_map],'--b');
% plot(p,MSE_map,'xb');
% 
% plot([1, 1],[MSE_obg, MSE_obg],'-g');
% plot([1, 1],[0, MSE_obg],'--g');
% plot(1, MSE_obg,'xg');
% 
% legend('MMSE','MAPC','DET','Oracle');
legend('MMSE','OBG','MAPC','DET','Oracle','Location','SouthWest');
xlabel('Computations (as a factor of p*d)','FontSize',14);
ylabel('RMSE','FontSize',14);
set(gca,'FontSize',12);
grid;
xlim([min(iter_mmse), sum(iter_mmse)]/(p*d));
print([filename '.eps'],'-deps2','-r600');
print([filename '.png'],'-dpng');

% figure;
% plot(1:max_iter, sum((XX-repmat(y,[1,max_iter])).^2),'Linewidth',2,'MarkerSize',8);
% xlabel('Iteration','FontSize',14);
% ylabel('MSE(y,x_{MMSE}^i)','FontSize',14);
% set(gca,'FontSize',12);
% grid;
% print([filename '-againstY.eps'],'-deps2','-r600');
% print([filename '-againstY.png'],'-dpng');

figure;
MSE_oracle2 = sum((x_oracle-y).^2);
denom = (d*sigma_n^2);
denom = sum((y-x).^2);
semilogx(cumsum(iter_mmse)/(p*d), sum((XX_mmse-repmat(y,[1,iter_mmse_l])).^2)./denom,'-k','Linewidth',2,'MarkerSize',8);
hold on;
plot(cumsum(iter_obg)/(p*d), sum((XX_obg-repmat(y,[1,iter_obg_l])).^2)./denom,'-r','Linewidth',2,'MarkerSize',8);
plot(cumsum(iter_map)/(p*d), sum((XX_map-repmat(y,[1,iter_map_l])).^2)./denom,'-b','Linewidth',2,'MarkerSize',8);
plot(cumsum(iter_det)/(p*d), sum((XX_det-repmat(y,[1,iter_det_l])).^2)./denom,'-g','Linewidth',2,'MarkerSize',8);
plot([min(iter_mmse), sum(iter_mmse)]/(p*d),[MSE_oracle2, MSE_oracle2]/denom,'--m');



end

