function createFigure1(O, q, sigmas, max_exper, filename, restore)

% Experiment setup

% Problem size
[p,d] = size(O);

% Model parameters
sigma_x = sqrt(1);
num_sigmas = numel(sigmas);

% Precompute the supports
[supps, ~] = CreateSupports(p);
In = eye(d);

% Precompute the extra terms for MMAP
[term_lambda, support_used] = PrecomputeMMAPTerms(O, q, supps, restore);

fprintf('Problem size p=%d, d=%d, q=%g\n',p,d,q);

% Experiment run
MSE_noise  = zeros(max_exper,num_sigmas);
MSE_oracle = zeros(max_exper,num_sigmas);
MSE_ora_th = zeros(max_exper,num_sigmas);
MSE_ora_lim = zeros(max_exper,num_sigmas);
MSE_mapex  = zeros(max_exper,num_sigmas);
MSE_rmapex = zeros(max_exper,num_sigmas);
MSE_mmseex = zeros(max_exper,num_sigmas);
MSE_mapth  = zeros(max_exper,num_sigmas);
MSE_mmseth = zeros(max_exper,num_sigmas);
MSE_rmapth = zeros(max_exper,num_sigmas);

for i = 1:num_sigmas
    sigma_n = sigmas(i);
    fprintf('Sigma=%g\n',sigma_n);
    parfor exper = 1:max_exper
%     for exper = 1:max_exper
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
        MSE_ora_th(exper,i) =  r*sigma_n^2*sigma_x^2/(sigma_x^2+sigma_n^2);
%         MSE_ora_lim(exper,i) =  r*sigma_n^2;
        
        [x_mmseex, x_mapex, x_rmapex, perf_mmse, perf_map, perf_rmap] = MMSEandMAPandMMAPAnalysisExhaustiveWithPerf(O, y, sigma_x, sigma_n, q, supps, support_used, term_lambda);
        MSE_mapex(exper,i)  = sum((x_mapex-x).^2);
        MSE_rmapex(exper,i) = sum((x_rmapex-x).^2);
        MSE_mmseex(exper,i) = sum((x_mmseex-x).^2);
        MSE_mapth(exper,i)  = perf_map;
        MSE_mmseth(exper,i) = perf_mmse;
        MSE_rmapth(exper,i) = perf_rmap;        
     end
end

clear term_lambda support_used supps

save(filename)

figure;
plot(sigmas, mean(MSE_oracle./MSE_noise) ,'sr','Linewidth',2);
hold on;
plot(sigmas, mean(MSE_mmseex./MSE_noise) ,'sg','Linewidth',2);
plot(sigmas, mean(MSE_rmapex./MSE_noise) ,'sb','Linewidth',2);
plot(sigmas, mean(MSE_mapex ./MSE_noise) ,'sc','Linewidth',2);
plot(sigmas, mean(MSE_ora_th./MSE_noise) ,'-r','Linewidth',2);
plot(sigmas, mean(MSE_mmseth./MSE_noise) ,'-g','Linewidth',2);
plot(sigmas, mean(MSE_rmapth./MSE_noise) ,'-b','Linewidth',2);
plot(sigmas, mean(MSE_mapth ./MSE_noise) ,'-c','Linewidth',2);
legend({'Oracle','MMSE Exahustive','RMAP Exahustive','MAP Exahustive', ...
    'Oracle Perf.','MMSE Perf.','RMAP Perf.','MAP Perf.'},'FontSize',12);
% axis([min(sigmas) max(sigmas) 0 1]);
xlim([min(sigmas) max(sigmas)]);
set(gca,'FontSize',12);
xlabel('\sigma_n','FontSize',14);
ylabel('Relative-Mean-Squared-Error','FontSize',14);
print([filename '-Perf.eps'],'-deps2','-r600');
print([filename '-Perf.png'],'-dpng');

figure;
plot(sigmas, mean(MSE_oracle./MSE_noise) ,'-r','Linewidth',2);
hold on;
plot(sigmas, mean(MSE_mmseex./MSE_noise) ,'-g','Linewidth',2);
plot(sigmas, mean(MSE_rmapex./MSE_noise) ,'-b','Linewidth',2);
plot(sigmas, mean(MSE_mapex ./MSE_noise) ,'-c','Linewidth',2);
legend({'Oracle','MMSE Exahustive','RMAP Exahustive','MAP Exahustive'},'FontSize',12);
% axis([min(sigmas) max(sigmas) 0 1]);
xlim([min(sigmas) max(sigmas)]);
set(gca,'FontSize',12);
xlabel('\sigma_n','FontSize',14);
ylabel('Relative-Mean-Squared-Error','FontSize',14);
print([filename '-noPerf.eps'],'-deps2','-r600');
print([filename '-noPerf.png'],'-dpng');

end
