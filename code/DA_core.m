function [Z,P] = DA_core(X_src,Y_src,X_tar,P,Y_tar_pseudo)
% The code is written by Mengcheng Lan,
% if you have any problems, please don't hesitate to contact me: lanmengchengds@gmail.com 
% If you find the code is useful, please cite the following reference:
% Min Meng, Mengcheng Lan, Jun Yu, Jigang Wu, 
% Coupled Knowledge Transfer for Visual Data Recognition [J], 
% IEEE Transactions on Circuits and Systems for Video Technology, 2020.

global options
%% Set options
alpha = options.alpha;                %% alpha for the trade-off
lambda = options.lambda;              %% lambda for the regularization
dim = options.dim;                    %% dim is the dimension after adaptation, dim <= m
kernel_type = options.kernel_type;    %% kernel_type is the kernel name, primal|linear|rbf
sigma = options.sigma;                %% gamma is the bandwidth of rbf kernel

%% intra-domain knowledge transfer
% Source Laplace matrix
Ws = Laplace_Simple_Guass(X_src,Y_src);
Ls = diag(sum(Ws)) - Ws;
Ls = Ls/norm(Ls,'fro');

% Y_tar_label = Y_tar_pseodu.*beta
[Y_tar_label] = TruePseudoLabel(X_src,X_tar,P,Y_tar_pseudo);
% Target Laplace matrix
Wt = Laplace_Simple_Guass(X_tar,Y_tar_label);
Lt = diag(sum(Wt)) - Wt;
Lt = Lt/norm(Lt,'fro');
L = blkdiag(Ls,Lt);

%% inter-domain distribution matching
%% Construct MMD matrix
X = [X_src X_tar];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
ns = size(X_src,2);
nt = size(X_tar,2);
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
C = length(unique(Y_src));
M = e * e' * C ;  %multiply C for better normalization

%% Construct CMMD matrix
n = size(X,2);
N = 0;
if ~isempty(Y_tar_label) && length(Y_tar_label)==nt
    for c = reshape(unique(Y_src),1,C)
        e = zeros(n,1);
        e(Y_src==c) = 1 / length(find(Y_src==c));
        e(ns+find(Y_tar_label==c)) = -1 / length(find(Y_tar_label==c));
        e(isinf(e)) = 0;
        N = N + e*e';
    end
end
M = M + N;
M = M / norm(M,'fro');

L_M = X*(L+alpha*M)*X';
L_M = X*(alpha*M)*X';

[m,n] = size(X);
H = eye(n) - 1/n * ones(n,n);

%% Calculation
if strcmp(kernel_type,'primal')
    [P,~] = eigs(L_M+lambda*eye(m),X*H*X',dim,'SM');
    Z = P'*X;
else
    K = kernel_jda(kernel_type,X,[],sigma);
    L_M = K*(L+alpha*M)*K';
    [P,~] = eigs(L_M+lambda*eye(n),K*H*K',dim,'SM');
    Z = P'*K;
end
end

% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf','sam'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the RBF/SAM kernel
% Output:
%       K: kernel matrix
%
% Gustavo Camps-Valls
% 2006(c)
% Jordi (jordi@uv.es), 2007
% 2007-11: if/then -> switch, and fixed RBF kernel
% Modified by Mingsheng Long
% 2013(c)
% Mingsheng Long (longmingsheng@gmail.com), 2013

function K = kernel_jda(ker,X,X2,gamma)

switch ker
    case 'linear'
        
        if isempty(X2)
            K = X'*X;
        else
            K = X'*X2;
        end
        
    case 'rbf'
        
        n1sq = sum(X.^2,1);
        n1 = size(X,2);
        
        if isempty(X2)
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end
        K = exp(-gamma*D);
        
    case 'sam'
        
        if isempty(X2)
            D = X'*X;
        else
            D = X'*X2;
        end
        K = exp(-gamma*acos(D).^2);
        
    otherwise
        error(['Unsupported kernel ' ker])
end
end