function [fTrain,fTest]=CSPfeature(xTrain,yTrain,xTest,nFilters)
%%  train CSP filters

if ~exist('nFilters','var') || isempty(nFilters)
    nFilters=3;
end

nChannels=size(xTrain,1);
cs=unique(yTrain);
xTrain0=xTrain(:,:,yTrain==cs(1));
xTrain1=xTrain(:,:,yTrain==cs(2));
Sigma0=zeros(nChannels);  Sigma1=zeros(nChannels);
for i=1:size(xTrain0,3)
    tmp0=cov(xTrain0(:,:,i)');
    Sigma0=Sigma0+tmp0;
end
for i=1:size(xTrain1,3)
    tmp1=cov(xTrain1(:,:,i)');
    Sigma1=Sigma1+tmp1;
end
Sigma0=Sigma0/size(xTrain0,3);
Sigma1=Sigma1/size(xTrain1,3);
[d,v]=eig(Sigma1\Sigma0);
[~,ids]=sort(diag(v),'descend');
W=d(:,ids([1:nFilters end-nFilters+1:end])); 

fTrain=zeros(size(xTrain,3),size(W,2));
fTest=zeros(size(xTest,3),size(W,2));
for i=1:size(xTrain,3)
    X=W'*xTrain(:,:,i);
    fTrain(i,:)=log10(diag(X*X')/trace(X*X'));
end
for i=1:size(xTest,3)
    X=W'*xTest(:,:,i);
    fTest(i,:)=log10(diag(X*X')/trace(X*X'));
end
