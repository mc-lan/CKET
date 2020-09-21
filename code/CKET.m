function [acc,acc_ite,P] = CKET(X_src,Y_src,X_tar,Y_tar)
% The code is written by Mengcheng Lan,
% if you have any problems, please don't hesitate to contact me: lanmengchengds@gmail.com 
% If you find the code is useful, please cite the following reference:
% Min Meng, Mengcheng Lan, Jun Yu, Jigang Wu, 
% Coupled Knowledge Transfer for Visual Data Recognition [J], 
% IEEE Transactions on Circuits and Systems for Video Technology, 2020.

% Inputs:
%%% X_src          :     source feature matrix, n_feature * ns
%%% Y_src          :     source label vector, ns * 1
%%% X_tar          :     target feature matrix, n_feature * nt
%%% Y_tar          :     target label vector, nt * 1
%%% options        :     option struct
%%%%% alpha        :     regularization parameter
%%%%% lambda       :     regularization parameter
%%%%% delta        :     neighborhood factor
%%%%% dim          :     dimension after adaptation, dim <= n_feature
%%%%% kernel_tpye  :     kernel name, choose from 'primal' | 'linear' | 'rbf'
%%%%% sigma        :     bandwidth for rbf kernel, can be missed for other kernels
%%%%% T            :     n_iterations, T >= 1. T <= 10 is suffice

% Outputs:
%%% acc            :     final accuracy using knn, float
%%% acc_ite        :     list of all accuracies during iterations
%%% P              :     final adaptation matrix, (ns + nt) * dim

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global options

% 1NN
acc_ite = [];
knn_model = fitcknn(X_src',Y_src,'NumNeighbors',1);
Y_tar_pseudo = knn_model.predict(X_tar');
acc = length(find(Y_tar_pseudo==Y_tar))/length(Y_tar);
fprintf('1NN=%0.4f\n',acc);
fprintf('******************************\n');
acc_ite = [acc_ite;acc];

P = [];
%% Iteration
for i = 1 : options.T
    %domain adaptive
    [Z,P] = DA_core(X_src,Y_src,X_tar,P,Y_tar_pseudo);
    
    %normalization for better classification performance
    Z = Z * diag(sparse(1./sqrt(sum(Z.^2))));
    Zs = Z(:,1:size(X_src,2));
    Zt = Z(:,size(X_src,2)+1:end);
    
    %calculation classication accuracy
    knn_model = fitcknn(Zs',Y_src,'NumNeighbors',1);
    Y_tar_pseudo = knn_model.predict(Zt');
    acc = length(find(Y_tar_pseudo==Y_tar))/length(Y_tar);
    fprintf('iter=%d\t',i);
    fprintf('CKET+1NN=%0.4f\n',acc);
    acc_ite = [acc_ite;acc];
end
end