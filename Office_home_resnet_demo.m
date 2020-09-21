clear;clear global;clc;
warning off;
global options
% Set algorithm parameters
options.alpha = 0.5;
options.lambda = 0.001;
options.dim = 100;  % subspace dimension
options.T = 10;  % iterations
options.delta = 0.05;  % neighborhood factor
options.sigma = 1;
options.kernel_type = 'primal';  % primal; linear; rbf;

srcStr = {'Art_Art','Art_Art','Art_Art','Clipart_Clipart','Clipart_Clipart','Clipart_Clipart',...
    'Product_Product','Product_Product','Product_Product','RealWorld_RealWorld','RealWorld_RealWorld','RealWorld_RealWorld'};
tgtStr = {'Art_Clipart','Art_Product','Art_RealWorld','Clipart_Art','Clipart_Product','Clipart_RealWorld',...
    'Product_Art','Product_Clipart','Product_RealWorld','RealWorld_Art','RealWorld_Clipart','RealWorld_Product'};

ffid = fopen('result_office_home_resnet.txt','at');
fprintf(ffid, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n %s\n', datestr(now));
fprintf(ffid, 'alpha = %.2f  lambda = %.2f  dim = %d \n', options.alpha,options.lambda,options.dim);
fprintf(ffid, 'delta = %.2f \t kernel = %s \n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n', options.delta,options.kernel_type);

addpath([pwd '/code'])
datapath = '.\data\Office-Home_resnet50\';
result = [];
for iData = 1:12
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    % Preprocess data using L2-norm
    load(fullfile(datapath,[src,'.mat']));
    X_src = normc(fts);
    X_src = zscore(X_src',1);
    X_src = normc(X_src');
    Y_src = labels;
    load(fullfile(datapath,[tgt,'.mat']));
    X_tar = normc(fts);
    X_tar = zscore(X_tar',1);
    X_tar = normc(X_tar');
    Y_tar = labels;
    
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