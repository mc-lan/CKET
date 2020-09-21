clear;clear global;clc;
warning off;
global options
% Set algorithm parameters
options.alpha = 5;
options.lambda = 0.01;
options.dim = 100;  % subspace dimension
options.T = 10; % iterations
options.delta = 0.25;  % neighborhood factor
options.sigma = 1;
options.kernel_type = 'primal';  % primal; linear; rbf;

srcStr = {'PIE05','PIE05','PIE05','PIE05','PIE07','PIE07','PIE07','PIE07','PIE09','PIE09','PIE09','PIE09','PIE27','PIE27','PIE27','PIE27','PIE29','PIE29','PIE29','PIE29'};
tgtStr = {'PIE07','PIE09','PIE27','PIE29','PIE05','PIE09','PIE27','PIE29','PIE05','PIE07','PIE27','PIE29','PIE05','PIE07','PIE09','PIE29','PIE05','PIE07','PIE09','PIE27'};

ffid = fopen('result_pie.txt','at');
fprintf(ffid, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n %s\n', datestr(now));
fprintf(ffid, 'alpha = %.2f  lambda = %.2f  dim = %d \n', options.alpha,options.lambda,options.dim);
fprintf(ffid, 'delta = %.2f \t kernel = %s \n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n', options.delta,options.kernel_type);

addpath([pwd '/code'])
datapath = '.\data\PIE\';
result = [];
for iData = 1:20
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    % Preprocess data using L2-norm
    load(fullfile(datapath, [src '.mat']));
    X_src = fea';
    X_src = normc(X_src);
    Y_src = gnd;
    load(fullfile(datapath, [tgt '.mat']));
    X_tar = fea';
    X_tar = normc(X_tar);
    Y_tar = gnd;
    
    fprintf('$$$$$$$$$$$$$$$ --%s-- $$$$$$$$$$$$$$\n' ,options.data);
    
    %%
    [acc,acc_ite,P] = CKET(X_src,Y_src,X_tar,Y_tar);
    result = [result acc];
    acc = 100*acc;
    fprintf('******************************\n%s :\naccuracy: %.2f\n\n',options.data,acc);
    fprintf(ffid,'***************************\n%s :\naccuracy: ',options.data);
    fprintf(ffid,'%.2f\n', acc);
end
fclose(ffid);