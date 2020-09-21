function [Y_tar_label] = TruePseudoLabel(X_src,X_tar,P,Y_tar_pseudo)
%Input
%X_src: the source data matrix, feature* number
%X_tar: the target data matrix, feature* number
%Y_tar_pseudo: the pseudo labels of target data, number*1
%P    : feature transformation
%Output
%Y_tar_label: number of target labels*1

global options

if options.delta==0
    Y_tar_label = Y_tar_pseudo;
else
    if strcmp(options.kernel_type,'primal')
        ;
    else
        X = [X_src X_tar];
        X = X*diag(sparse(1./sqrt(sum(X.^2))));
        K = kernel_jda(options.kernel_type,X,[],options.sigma);
        X_tar = K(:,size(X_src,2)+1:end);
    end
    if ~isempty(P)
        PX_tar = P'*X_tar;
    else
        PX_tar = X_tar;
    end
    numOfclasses = unique(Y_tar_pseudo);
    classNum = zeros(length(numOfclasses),1);
    k = zeros(length(numOfclasses),1);
    for c=1:length(numOfclasses)
        classNum(numOfclasses(c)) = length(find(Y_tar_pseudo==numOfclasses(c)));
        k(numOfclasses(c)) = ceil(classNum(numOfclasses(c))*options.delta);
    end
    k_max = max(k);
    neighs = knnsearch(PX_tar',PX_tar','k',k_max,'distance','euclidean');
    Y_tar_label = zeros(size(X_tar,2),1); %beta = 0
    for i=1:size(X_tar,2)
        c = Y_tar_pseudo(i);
        index = neighs(i,1:k(c));
        if sum(Y_tar_pseudo(i)==Y_tar_pseudo(index))==k(c)   %formula (9)
            Y_tar_label(i,1) = Y_tar_pseudo(i); %beta_i = 1
        end
    end
end
end

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