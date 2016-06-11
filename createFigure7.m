function createFigure7(O, q, sigmas, max_exper, filename, regularizers, max_iter)

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
MSE_map  = zeros(max_exper,num_sigmas);
MSE_mmse = zeros(max_exper,num_sigmas);
MSE_det = zeros(max_exper,num_sigmas);
MSE_detLam = zeros(max_exper,num_sigmas,numel(regularizers ));

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
        
        % MMSE approx
        [x_mmse, ~] = MMSEgibbs_temp(O, y, sigma_x, sigma_n, q, max_iter);
        MSE_mmse(exper,i) = sum((x_mmse-x).^2);
        
        % DET approx
        [x_det] = DET2AnalysisErr2_temp(O, y, sigma_x, sigma_n, q);
        MSE_det(exper,i) = sum((x_det-x).^2);        
        
        % MAP approx
        x_map = MAP2AnalysisErr2_temp(O, y, sigma_x, sigma_n, q);
        MSE_map(exper,i) = sum((x_map-x).^2);
        
        % Run DET with different values
        dd = zeros(numel(regularizers),1);
        for l = 1:numel(regularizers)
%             x_detR2 = DETAnalysisErr2lambda(O, y, sigma_x, sigma_n, regularizers(l));
            x_detR = DET2AnalysisErr2lambda_temp(O, y, sigma_x, sigma_n, regularizers(l)); % faster
            dd(l) = sum((x_detR-x).^2);
%             disp(sum((x_detR-x_detR2).^2));
        end
        MSE_detLam(exper,i,:) = dd;
     end
end

save(filename)

% Plot of the results
figure;
% Exhaustive
plot(sigmas, mean(MSE_oracle./MSE_noise) ,'-r','Linewidth',2);
hold on;
plot(sigmas, mean(MSE_mmse./MSE_noise),'x-k','Linewidth',2,'MarkerSize',8);
plot(sigmas, mean(MSE_map./MSE_noise) ,'x-g','Linewidth',2,'MarkerSize',8);
plot(sigmas, mean(MSE_det./MSE_noise) ,'x-m','Linewidth',2,'MarkerSize',8);

% Deterministic
% for i = 1:numel(regularizers)
%     z =  mean(MSE_detLam(:,:,i)./MSE_noise);
%     plot(sigmas, z ,'-b');
% %     disp(regularizers(i));
% %     pause
% %     text(sigmas(2),z(2),[' \leftarrow + ' num2str(regularizers(i))]);
% end
plot(sigmas, min(mean(MSE_detLam./repmat(MSE_noise,[1,1,numel(regularizers)])),[],3),'s-b','Linewidth',2,'MarkerSize',8);

% repeat the plot to make it visible over the blue lines
% plot(sigmas, mean(MSE_mmse./MSE_noise),'x-k','Linewidth',2,'MarkerSize',8);
% plot(sigmas, mean(MSE_map./MSE_noise) ,'x-g','Linewidth',2,'MarkerSize',8);
% plot(sigmas, mean(MSE_det./MSE_noise) ,'x-m','Linewidth',2,'MarkerSize',8);


legend({'Oracle','MMSE Approx','MAPC Approx','DET Approx Precomp \lambda_0','DET Approx Best \lambda_0'},'FontSize',12);
xlabel('\sigma_n','FontSize',14);
ylabel('Relative-Mean-Squared-Error','FontSize',14);
% axis([min(sigmas) max(sigmas) 0 1]);
xlim([min(sigmas) max(sigmas)]);
set(gca,'FontSize',12);
print([filename '.eps'],'-deps2','-r600');
print([filename '.png'],'-dpng');


end