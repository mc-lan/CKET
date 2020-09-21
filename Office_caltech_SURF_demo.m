clear;clear global;clc;
warning off;
global options
% Set algorithm parameters
options.alpha = 0.5;
options.lambda = 0.1;
options.dim = 10;  % subspace dimension
options.T = 10;  % iterations
options.delta = 0;  % neighborhood factor
options.sigma = 1;
options.kernel_type = 'primal';  % primal; linear; rbf;

srcStr = {'amazon','amazon','amazon','Caltech10','Caltech10','Caltech10','dslr','dslr','dslr','webcam','webcam','webcam'};
tgtStr = {'Caltech10','dslr','webcam','amazon','dslr','webcam','amazon','Caltech10','webcam','amazon','Caltech10','dslr'};

ffid = fopen('result_office_caltech_SURF.txt','at');
fprintf(ffid, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n %s\n', datestr(now));
fprintf(ffid, 'alpha = %.2f  lambda = %.2f  dim = %d \n', options.alpha,options.lambda,options.dim);
fprintf(ffid, 'delta = %.2f \t kernel = %s \n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n', options.delta,options.kernel_type);

addpath([pwd '/code'])
datapath = '.\data\Office+Caltech+surf\';
result = [];
for iData = 1:12
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    % Preprocess data using L2-norm
    load(fullfile(datapath,[src,'_zscore_SURF_L10.mat']));
    X_src = Xt';
    X_src = normc(X_src);
    Y_src = Yt;
    load(fullfile(datapath,[tgt,'_zscore_SURF_L10.mat']));
    X_tar = Xt';
    X_tar = normc(X_tar);
    Y_tar = Yt;
    
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