function [fTrain,fTest]=RCSPfeature(Xs,ys,Xt,ytTrain,nFilters,beta,gamma)
%%  train RCSP filters

if ~exist('nFilters','var') || isempty(nFilters)
    nFilters=3;
end

if ~exist('beta','var') || isempty(beta)
    beta=0.1;
end

if ~exist('gamma','var') || isempty(gamma)
    gamma=0.1;
end

[nChannels,~,N]=size(Xs); M=size(Xt,3); m=length(ytTrain);
XtTest=Xt(:,:,m+1:end);
cs=unique(ys);
Xs0=Xs(:,:,ys==cs(1)); Xs1=Xs(:,:,ys==cs(2));
XtTrain0=Xt(:,:,find(ytTrain==cs(1)));
XtTrain1=Xt(:,:,find(ytTrain==cs(2)));
SigmaS0=zeros(nChannels);  SigmaS1=zeros(nChannels);
SigmaT0=zeros(nChannels);  SigmaT1=zeros(nChannels);
for i=1:size(Xs0,3)
    SigmaS0=SigmaS0+cov(Xs0(:,:,i)');
end
for i=1:size(Xs1,3)
    SigmaS1=SigmaS1+cov(Xs1(:,:,i)');
end
for i=1:size(XtTrain0,3)
    SigmaT0=SigmaT0+cov(XtTrain0(:,:,i)');
end
for i=1:size(XtTrain1,3)
    SigmaT1=SigmaT1+cov(XtTrain1(:,:,i)');
end
Omega0=((1-beta)*SigmaT0+beta*SigmaS0)/((1-beta)*size(XtTrain0,3)+beta*size(Xs0,3));
Omega1=((1-beta)*SigmaT1+beta*SigmaS1)/((1-beta)*size(XtTrain1,3)+beta*size(Xs1,3));
SigmaST0=(1-gamma)*Omega0+gamma*trace(Omega0)*eye(nChannels)/nChannels;
SigmaST1=(1-gamma)*Omega1+gamma*trace(Omega1)*eye(nChannels)/nChannels;
[d,v]=eig(SigmaST1\SigmaST0);
[~,ids]=sort(diag(v),'descend');
W=d(:,ids([1:nFilters end-nFilters+1:end])); 

XTrain=cat(3,Xt(:,:,1:m),Xs);
fTrain=zeros(N+m,2*nFilters);
fTest=zeros(M-m,2*nFilters);
for i=1:N+m
    X=W'*XTrain(:,:,i);
    fTrain(i,:)=log10(diag(X*X')/trace(X*X'));
end
for i=1:M-m
    X=W'*XtTest(:,:,i);
    fTest(i,:)=log10(diag(X*X')/trace(X*X'));
end
