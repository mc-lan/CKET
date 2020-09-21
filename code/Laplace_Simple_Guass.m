function W=Laplace_Simple_Guass(X_train,Y_train)
sigma = 5;
[~,col]=size(X_train);
W=zeros(col);
for i=1:col
    for j=1:col
        if Y_train(i)==Y_train(j)&&Y_train(i)~=0
            W(i,j)=exp(-norm(X_train(:,i)-X_train(:,j),2)^2/sigma);
        end
    end
end
end