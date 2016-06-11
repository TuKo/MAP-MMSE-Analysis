function createFigure5(O, q, sigmas, max_exper, filename, max_iter)

% Experiment setup

% Problem size
[p,d] = size(O);

% Model parameters
sigma_x = sqrt(1);
num_sigmas = numel(sigmas);

In = eye(d);

fprintf('Problem size p=%d, d=%d, q=%g\n',p,d,q);

% Experiment run
MSE_noise  = zeros(max_exper,num_sigmas);
MSE_oracle = zeros(max_exper,num_sigmas);
% MSE_map    = zeros(max_exper,num_sigmas);
MSE_mmse   = zeros(max_exper,num_sigmas);
MSE_det    = zeros(max_exper,num_sigmas);

for i = 1:num_sigmas
    sigma_n = sigmas(i);
    fprintf('Sigma=%g\n',sigma_n);
    parfor exper = 1:max_exper
%     for exper = 1:max_exper, 
%         fprintf('exper=%d\n',exper);
        OO = O; % this is for parallelizing the for
        % Draw a noisy signal.
        cosupp = rand(p,1) < q;
        cosupp_size = sum(cosupp);
        if (cosupp_size > 0)
            Os = OO(cosupp,:);
            Os = orth(Os'); % Use orthogonalization to avoid zero columns.
            Q = In - Os*Os';
            r = d-size(Os,2);
        else
            Q = In;
            r = d;
        end
        
        x0 = randn(d,1) * sigma_x;
        x = Q * x0;

        y = x + sigma_n*randn(d,1);        

        MSE_noise(exper,i) = d*sigma_n^2; % exact

        x_oracle = OracleAnalysis(O,y,sigma_x,sigma_n,cosupp);
        MSE_oracle(exper,i) = sum((x_oracle-x).^2);

%         C = Q + sigma_n^2/sigma_x^2*In;
%         MSE_ora_th(exper,i) =  sigma_n^2*trace((C\Q));
%         MSE_ora_th(exper,i) =  r*sigma_n^2*sigma_x^2/(sigma_x^2+sigma_n^2);
%         MSE_ora_lim(exper,i) =  r*sigma_n^2;
       
        % MMSE approx
        [x_mmse, ~] = MMSEgibbs_temp(O, y, sigma_x, sigma_n, q, max_iter);
        MSE_mmse(exper,i) = sum((x_mmse-x).^2);

        % DET approx
        [x_det] = DET2AnalysisErr2_temp(O, y, sigma_x, sigma_n, q);
%         [x_det2] = DETAnalysisErr2(O, y, sigma_x, sigma_n, q);
%         disp(sum((x_det2-x_det).^2));
        MSE_det(exper,i) = sum((x_det-x).^2);
        
     end
end

save(filename)

figure;
plot(sigmas, mean(MSE_oracle./MSE_noise) ,'-r','Linewidth',2);
hold on;
plot(sigmas, mean(MSE_mmse  ./MSE_noise) ,'-g','Linewidth',2);
plot(sigmas, mean(MSE_det   ./MSE_noise) ,'-k','Linewidth',2);
legend({'Oracle','MMSE Approx.','DET Approx.'},'FontSize',12);
% axis([min(sigmas) max(sigmas) 0 1]);
xlim([min(sigmas) max(sigmas)]);
set(gca,'FontSize',12);
xlabel('\sigma_n','FontSize',14);
ylabel('Relative-Mean-Squared-Error','FontSize',14);
print([filename '.eps'],'-deps2','-r600');
print([filename '.png'],'-dpng');

end
