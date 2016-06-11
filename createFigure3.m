function createFigure3(O, q, sigmas, max_exper, filename, regularizers, restore)

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
MSE_mapex  = zeros(max_exper,num_sigmas);
MSE_mmseex = zeros(max_exper,num_sigmas);
MSE_mapdetex = zeros(max_exper,num_sigmas);
MSE_rmapex = zeros(max_exper,num_sigmas);
MSE_detex = zeros(max_exper,num_sigmas, numel(regularizers ));

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
%             r = d-size(Os,2);
        else
            Q = In;
%             r = d;
        end
        
        x0 = randn(d,1) * sigma_x;
        x = Q * x0;

        y = x + sigma_n*randn(d,1);        

        MSE_noise(exper,i) = d*sigma_n^2; % exact

        x_oracle = OracleAnalysis(O,y,sigma_x,sigma_n,cosupp);
        MSE_oracle(exper,i) = sum((x_oracle-x).^2);
        
        [x_mmseex, x_mapex, x_det, x_rmapex] = MMSEandMAPandDETAnalysisExhaustive(O, y, sigma_x, sigma_n, q, supps, support_used, term_lambda);
        MSE_mapex(exper,i)  = sum((x_mapex-x).^2);
        MSE_rmapex(exper,i) = sum((x_rmapex-x).^2);
        MSE_mmseex(exper,i) = sum((x_mmseex-x).^2);
        MSE_mapdetex(exper,i) = sum((x_det-x).^2);
        
        x_detR = DETAnalysisExhaustiveLambdas(O, y, sigma_x, sigma_n, supps, regularizers);
        MSE_detex(exper,i,:) = sum((x_detR-repmat(x,[1,numel(regularizers)])).^2);
     end
end

clear term_lambda support_used supps

save(filename)

% Plot of the results
figure;
% Exhaustive
plot(sigmas, mean(MSE_oracle./MSE_noise) ,'-r','Linewidth',2);
hold on;
% plot(sigmas, mean(MSE_mapex./MSE_noise)   ,'s-b','Linewidth',2,'MarkerSize',10);
plot(sigmas, mean(MSE_rmapex./MSE_noise)  ,'d-k','Linewidth',2,'MarkerSize',8);
plot(sigmas, mean(MSE_mapdetex./MSE_noise),'o-m','Linewidth',2,'MarkerSize',8);
% plot(sigmas, mean(MSE_mmseex./MSE_noise)  ,'s-k','Linewidth',2,'MarkerSize',10);

% Deterministic
% for i = 1:numel(regularizers)
%     z =  mean(MSE_detex(:,:,i)./MSE_noise);
%     plot(sigmas, z ,'-b');
% %     text(sigmas(2),z(2),[' \leftarrow + ' num2str(regularizers(i))]);
% end
plot(sigmas, min(mean(MSE_detex(:,:,:)./repmat(MSE_noise,[1,1,numel(regularizers)])),[],3),'s-b','Linewidth',2,'MarkerSize',8);

% % repeat the plot to make it visible over the blue lines
% plot(sigmas, mean(MSE_rmapex./MSE_noise)  ,'x-g','Linewidth',2,'MarkerSize',8);
% plot(sigmas, mean(MSE_mapdetex./MSE_noise),'x-m','Linewidth',2,'MarkerSize',8);


legend({'Oracle','MAPS exhaustive','DET ex. precomp. \lambda_0','DET ex. best \lambda_0'},'FontSize',12);
xlabel('\sigma_n','FontSize',14);
ylabel('Relative-Mean-Squared-Error','FontSize',14);
% axis([min(sigmas) max(sigmas) 0.1 0.55]);
xlim([min(sigmas) max(sigmas)]);
set(gca,'FontSize',12);
print([filename '.eps'],'-deps2','-r600');
print([filename '.png'],'-dpng');


end