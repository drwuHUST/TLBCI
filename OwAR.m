function [yt1,f,alpha,K]=OwAR(Xs,ys,Xt,yt0,options,K)
%% Online weighted adaptation regularization (OwAR) for online calibration (2-class classification)
%% Dongrui Wu, drwu09@gmail.com

%% Inputs:
%%  Xs: Features in source domain
%%  ys: column vector; labels in source domain
%%  Xt: Features in target domain. The first ml=length(yt0) rows are epochs
%%      with known labels, and the rest rows are (unknown) testing epochs
%%  yt0: column vector; known labels in target domain
%%       (corresponding to the first ml=length(yt0) rows of Xt)
%%  options: optional regularization parameters
%%      sigma: regularization for structural risk; default .1
%%      lambda: regularization for marginal and conditional probability distributions; default 10
%%      wt: overall weight for the target domain samples; default 2
%%  K: Optional, Kernal matrix for [Xs; Xt]; avoid computing K every time

%% Outputs:
%%  yt1: column vector; estimated labels for the last size(Xt,1)-length(yt0) rows of Xt
%%  f: weighted training accuracy on Xs and the first ml rows of Xt; used as weight in OWARSDS
%%  alpha: parameter alpha in the classifier
%%  K: Kernal matrix for [Xs; Xt]

if nargin<5; options=[]; end

if ~isfield(options,'sigma');     options.sigma=0.1;  end % weight on structural risk
if ~isfield(options,'lambda');    options.lambda=10; end % weight on probabilities
if ~isfield(options,'wt');        options.wt=2.0; end % overall weight for target domain samples

sigma=options.sigma;  lambda=options.lambda;  wt=options.wt;

%% Initialization
X=[Xs; Xt];
Y=[ys; yt0];
Cs=unique(Y);
n=length(ys); Ws=ones(n,1);
Ws(ys==Cs(2))=sum(ys==Cs(1))/sum(ys==Cs(2)); % weight for minority class in source domain
ml=length(yt0); Wt=ones(ml,1);
Wt(yt0==Cs(2))=sum(yt0==Cs(1))/sum(yt0==Cs(2)); % weight for minority class in target domain

% %% Data normalization: Make the squared sum of each feature vector 1
% X=diag(sparse(1./sqrt(sum(X.^2,2))))*X;

%% Construct MMD matrix
e=[1/n*ones(n,1); -1/ml*ones(ml,1)];
M=e*e'*length(Cs);
for c=Cs'
    e=zeros(n+ml,1);
    e(ys(1:n)==c)=1/sum(ys==c);
    e(n+find(yt0==c))=-1/sum(yt0==c);
    e(isinf(e))=0;
    M=M+e*e';
end
%M=M/norm(M,'fro');

%% Compute K; linear kernel; more complex kernels could be used
if nargin<6 || isempty(K)
    K=X*X';
end

%% Compute alpha
W=[Ws; wt*Wt]; E=diag(W);
alpha=((E+lambda*M)*K(1:n+ml,1:n+ml)+sigma*eye(n+ml,n+ml))\(E*Y);

%% Classification
yt1Raw=K(n+ml+1:end,1:n+ml)*alpha;
yt1=sign(yt1Raw); ids1=yt1==1;
yt1(~ids1)=Cs(1); yt1(ids1)=Cs(2);

%% Compute weight for the classifier
%f=norm(E*(Y-K*alpha))+trace(sigma*alpha'*K*alpha+alpha'*K*(lambda*M+gamma*L)*K*alpha);
yRaw=sign(K(1:n+ml,1:n+ml)*alpha);
ids1=yRaw==1;
yRaw(~ids1)=Cs(1); yRaw(ids1)=Cs(2);
f=(yRaw==Y)'*W/sum(W);

