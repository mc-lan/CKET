clear;clear global;clc;
warning off;
global options
% Set algorithm parameters
options.alpha = 0.5;
options.lambda = 0.05;
options.dim = 10;  % subspace dimension
options.T = 10;  % iterations
options.delta = 0.35;  % neighborhood factor
options.sigma = 1;
options.kernel_type = 'primal';  % primal; linear; rbf;

srcStr = {'amazon','amazon','amazon','caltech','caltech','caltech','dslr','dslr','dslr','webcam','webcam','webcam'};
tgtStr = {'caltech','dslr','webcam','amazon','dslr','webcam','amazon','caltech','webcam','amazon','caltech','dslr'};

ffid = fopen('result_office_caltech_decaf6.txt','at');
fprintf(ffid, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n %s\n', datestr(now));
fprintf(ffid, 'alpha = %.2f  lambda = %.2f  dim = %d \n', options.alpha,options.lambda,options.dim);
fprintf(ffid, 'delta = %.2f \t kernel = %s \n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n', options.delta,options.kernel_type);

addpath([pwd '/code'])
datapath = '.\data\Office+Caltech+decaf6\';
result = [];
for iData = 1:12
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    % Preprocess data using L2-norm
    load(fullfile(datapath,[src,'_decaf.mat']));
    X_src = feas';
    X_src = normc(X_src);
    Y_src = labels;
    load(fullfile(datapath,[tgt,'_decaf.mat']));
    X_tar = feas';
    X_tar = normc(X_tar);
    Y_tar = labels;
    
    fprintf('$$$$$$$$$$$$$$$ --%s-- $$$$$$$$$$$$$$\n' ,options.data);
    
    %%
    [acc,acc_ite,P] = CKET(X_src,Y_src,X_tar,Y_tar);
    result = [result acc];
    acc = 100*acc;
    fprintf('******************************\n%s :\naccuracy: %.2f\n\n',options.data,acc);
    fprintf(ffid,'******************************\n%s :\naccuracy: ',options.data);
    fprintf(ffid,'%.2f\n', acc);
end
fclose(ffid);