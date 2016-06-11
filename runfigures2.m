%% Setup

% matlabpool 8
matlabpool 2
% matlabpool close

folder = 'figures/';
restore = 'TV';

%% Figure 1: Exhaustive estimators with performance measurements.
close all;

% Create a Derivatives dictionary (normalized)
n1 = 4;
n2 = 3;
Odiff = full(OperatorG(n1,n2));
[p,d] = size(Odiff);

% sigmas = 0.1:0.1:1.0; % possible sigmas for the noise
sigmas = [0.01:0.03:0.07,0.1:0.1:1.0]; % possible sigmas for the noise
max_exper = 1000; % Number of signals per experiment

i = 10;
q = i/p;
tic
filename = [folder 'figure1-d' num2str(d) 'p' num2str(p) 'q' num2str(i)];
createFigure1(Odiff, q, sigmas, max_exper, filename, restore);
toc

%% Figure 2: (similar to Fig 1) Show that DET and RMAP have similar 
% behaviours.
close all;

% Use same set-up as Figure 1.
tic
filename = [folder 'figure2-d' num2str(d) 'p' num2str(p) 'q' num2str(i)];
createFigure2(Odiff, q, sigmas, max_exper, filename, restore);
toc

%% Figure 3: Deterministic approach with distinct const values for the 
% regularizer lambda_0 showing that the linearly independent case is quite
% good.
close all;

% Create a Derivatives dictionary (normalized)
n1 = 4;
n2 = 3;
Odiff = full(OperatorG(n1,n2));
[p,d] = size(Odiff);

sigmas = [0.01:0.03:0.07,0.1:0.1:1.0]; % possible sigmas for the noise
% sigmas = 0.1:0.1:1; % possible sigmas for the noise
max_exper = 1000; % Number of signals per experiment

i = 10;
q = i/p;
regularizers = 0:0.1:3;
tic
filename = [folder 'figure3-d' num2str(d) 'p' num2str(p) 'q' num2str(i)];
createFigure3(Odiff, q, sigmas, max_exper, filename, regularizers', restore);
toc

%% Figure 4: Exhaustive estimators and their approximations.
% (add performance measurements?)
close all;

% Create a Derivatives dictionary (normalized)
n1 = 4;
n2 = 3;
Odiff = full(OperatorG(n1,n2));
[p,d] = size(Odiff);

sigmas = [0.01,0.05,0.1:0.1:1.0]; % possible sigmas for the noise
max_exper = 1000; % Number of signals per experiment
max_iter = 400;

i = 10;
q = i/p;
tic
filename = [folder 'figure4-d' num2str(d) 'p' num2str(p) 'q' num2str(i)];
createFigure4(Odiff, q, sigmas, max_exper, filename, max_iter, restore);
toc

%% Figure 5: Approximations comparison
close all;

% Create a Derivatives dictionary (normalized)
n1 = 8;
n2 = 8;
Odiff = full(OperatorG(n1,n2));
[p,d] = size(Odiff);

sigmas = [0.01,0.05,0.1:0.1:1.0]; % possible sigmas for the noise
max_exper = 1000; % Number of signals per experiment
max_iter = 5000; % Number of iterations for Gibbs sampler

% max_exper = 100; % Number of signals per experiment
% max_iter = 200;
% sigmas = [0.01,0.05,0.1];

i = 70;
q = i/p;
tic
filename = [folder 'figure5-d' num2str(d) 'p' num2str(p) 'q' num2str(i)];
createFigure5(Odiff, q, sigmas, max_exper, filename, max_iter);
toc

i = 60;
q = i/p;
tic
filename = [folder 'figure5-d' num2str(d) 'p' num2str(p) 'q' num2str(i)];
createFigure5(Odiff, q, sigmas, max_exper, filename, max_iter);
toc

i = 90;
q = i/p;
tic
filename = [folder 'figure5-d' num2str(d) 'p' num2str(p) 'q' num2str(i)];
createFigure5(Odiff, q, sigmas, max_exper, filename, max_iter);
toc



%% Figure 6: Convergence rate for an approximation of all the algorithms

% close all;

% Create a Derivatives dictionary (normalized)
% n1 = 8; n2 = 8; i = 80;
n1 = 12; n2 = 12; i = 160;
O = full(OperatorG(n1,n2));
[p,d] = size(O);

% sigma_n = 0.1;
sigma_n=0.2;
max_iter = 5000;

% max_exper = 1000;
% i = 60;

q = i/p;
tic
filename = [folder 'figure6-d' num2str(d) 'p' num2str(p) 'q' num2str(i)];
createFigure6(O, q, sigma_n, filename, max_iter);
toc
% tic
% filename = [folder 'figure6b-d' num2str(d) 'p' num2str(p)];
% q = 0.1:0.1:0.9;
% createFigure6b(Odiff, q, sigma_n, filename, max_iter,max_exper);
% toc

%% Figure 7: Deterministic approximation with distinct const values for the 
% regularizer lambda_0 showing that the linearly independent case is quite
% good (similar to Figure 3, but is not exhaustive).
close all;

% Create a Derivatives dictionary (normalized)
n1 = 8;
n2 = 8;
Odiff = full(OperatorG(n1,n2));
[p,d] = size(Odiff);

sigmas = [0.01:0.03:0.07,0.1:0.1:1.0]; % possible sigmas for the noise
max_exper = 1000; % Number of signals per experiment
max_iter = 5000; % Number of iterations for Gibbs sampler
regularizers = 0.8:0.1:2.3;

% sigmas = 0.1:0.1:0.2; % possible sigmas for the noise
% max_exper = 8; % Number of signals per experiment
% regularizers = 1:0.1:1.8;

% max_exper = 10; % Number of signals per experiment
% max_iter = 200;
% regularizers = 0.5:0.1:1;

i = 70;
q = i/p;
tic
filename = [folder 'figure7x-d' num2str(d) 'p' num2str(p) 'q' num2str(i)];
createFigure7(Odiff, q, sigmas, max_exper, filename, regularizers', max_iter);
toc
% i = 80;
% q = i/p;
% tic
% filename = [folder 'figure7x-d' num2str(d) 'p' num2str(p) 'q' num2str(i)];
% createFigure7(Odiff, q, sigmas, max_exper, filename, regularizers', max_iter);
% toc

