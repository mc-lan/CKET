clear;clear global;clc;
warning off;
global options
% Set algorithm parameters
options.alpha = 0.5;
options.lambda = 0.01;
options.dim = 10;  % subspace dimension
options.T = 10;  % iterations
options.delta = 0.25;  % neighborhood factor
options.sigma = 1;
options.kernel_type = 'primal';

srcStr = {'amazon','amazon','amazon','caltech','caltech','caltech','dslr','dslr','dslr','webcam','webcam','webcam'};
tgtStr = {'caltech','dslr','webcam','amazon','dslr','webcam','amazon','caltech','webcam','amazon','caltech','dslr'};

addpath([pwd '/code'])
datapath = '.\data\Office+Caltech+vgg\';

ffid = fopen('result_office_caltech_vgg-split.txt','at');
fprintf(ffid, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n %s\n', datestr(now));
fprintf(ffid, 'alpha = %.2f  lambda = %.2f  dim = %d \n', options.alpha,options.lambda,options.dim);
fprintf(ffid, 'delta = %.2f \t kernel = %s \n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n', options.delta,options.kernel_type);

results = [];
for iData = 1:12
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'-vs-',tgt);
    fprintf('Data=%s \n',options.data);
    % Preprocess data using L2-norm
    load(fullfile(datapath,[src,'_VGG-FC6.mat']));
    X_src = FTS';
    X_src = normc(X_src);
    Y_src = LABELS';
    load(fullfile(datapath,[tgt,'_VGG-FC6.mat']));
    X_tar = FTS';
    X_tar = normc(X_tar);
    Y_tar = LABELS';
    
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    splitdata = ['DataSplitsOfficeCaltech\SameCategory_', src, '-', tgt, '_20RandomTrials_10Categories.mat'];
    load(fullfile(datapath,splitdata));
    acc = zeros(length(train.source),1);
    for sp = 1: length(train.source)
        inds = train.source{sp};
        nXs = X_src(:,inds);
        nYs = Y_src(inds);
        fprintf('%d iterions\n',sp);
        [sp_acc,acc_ite,P] = CKET(nXs,nYs,X_tar,Y_tar);
        acc(sp,:) = 100*sp_acc;
    end
    
    macc = mean(acc);
    sacc = std(acc);
    results = [results; macc];

    fprintf('%s : \n NN mean accuracy: %.2f + %.2f\n***************************\n',options.data, macc, sacc);
    fprintf(ffid,'%s : \n NN mean accuracy: %.2f + %.2f\n***************************\n',options.data, macc, sacc);
end
fclose(ffid);