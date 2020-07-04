function [yt1,yRaw,f,alpha,K]=wAR(Xs,ys,Xt,yt0,yt1,options,K)
%% Weighted adaptation regularization (wAR) for offline calibration (2-class classification)
%% D. Wu*, "Online and Offline Domain Adaptation for Reducing BCI Calibration Effort," 
%% IEEE Trans. on Human-Machine Systems, vol. 47, no. 4, pp. 550-563, 2017.
%% Dongrui Wu, drwu09@gmail.com
%
%% Inputs:
%%  Xs: Features in source domain
%%  ys: column vector; labels in source domain; the two classes must be labeled as -1 and 1
%%  Xt: Features in target domain. The first ml=length(yt0) rows are epochs
%%      with known labels, and the rest rows are (unknown) testing epochs
%%  yt0: column vector; known labels in target domain; the two classes must be labeled as -1 and 1
%%       (corresponding to the first ml=length(yt0) rows of Xt)
%%  yt1: column vector; pseudo-labels in target domain
%%       (corresponding to the last size(Xt,1)-length(yt0) rows of Xt)
%%       yt1 can be obtained from the previous iteration of WAR;
%%       if yt1=[] or not supplied, then we will use weighted libSVM to estimate it
%%  options: optional regularization parameters
%%      sigma: regularization for structural risk; default .1
%%      lambda: regularization for marginal and conditional probability distributions; default 10
%%      wt: overall weight for the target domain samples; default 2
%%  K: Optional, Kernal matrix for [Xs; Xt]; avoid computing K every time
%
%% Outputs:
%%  yt1: column vector; estimated labels for the last size(Xt,1)-length(yt0) rows of Xt
%%  f: weighted training accuracy on Xs and the first ml rows of Xt; used as weight in wARSDS
%%  alpha: parameter alpha in the classifier
%%  K: Kernal matrix for [Xs; Xt]

if nargin<6; options=[]; end

if ~isfield(options,'sigma');     options.sigma=0.1;  end % weight on structural risk
if ~isfield(options,'lambda');    options.lambda=10; end % weight on probabilities
if ~isfield(options,'wt');        options.wt=10; end % overall weight for target domain samples

sigma=options.sigma; lambda=options.lambda;  wt=options.wt;

%% Compute yt1, the pseudo-labels, if it is not available
Cs=unique(cat(1,ys,yt0));
n=length(ys); m=size(Xt,1);
w=sum(ys==Cs(1))/sum(ys==Cs(2)); % weight for minority class in the source domain
Ws=ones(n,1); Ws(ys==Cs(2))=w;
%% Uncomment the following if libsvm-weight is not in your path
addpath('D:\Matlab2020a\toolbox\libsvm-weights-3.24\matlab');
% yt0 empty means the first iteration of WAR; need to estimate yt1 from ys
if nargin<5 || isempty(yt1)
    if nargin<4; yt0=[]; end
    y=[ys; yt0];  ml=length(yt0); X=[Xs; Xt(1:ml,:)]; maxP=0;
    Wt=ones(ml,1); Wt(yt0==Cs(2))=sum(yt0==Cs(1))/sum(yt0==Cs(2)); W=[Ws; Wt];
    %     model=fitcsvm(X,y,'weights',W,'Standardize',true,'KernelFunction','RBF','OptimizeHyperparameters','auto',...
    %     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    %     'expected-improvement-plus'));
    for log2c=-1:5 % SVM, 5-fold CV
        C=2^log2c;
        for log2g=-4:2
            Gamma=2^log2g;
            P=svmtrain(W,y,X,['-h 0 -v 5 -c ' num2str(C) ' -g ' num2str(Gamma)]);
            if P>maxP;       maxP=P; bestC=C; bestG=Gamma;       end
        end
    end
    model=svmtrain(W,y,X,['-h 0 -c ' num2str(bestC) ' -g ' num2str(bestG)]);
    yt1=svmpredict(ones(m-ml,1),Xt(ml+1:end,:),model);
else % yt1 available; use it directly in following computations
    ml=length(yt0); Wt=ones(ml,1); Wt(yt0==Cs(2))=sum(yt0==Cs(1))/sum(yt0==Cs(2));
end
yt=cat(1, yt0, yt1); X=cat(1, Xs, Xt); Y=cat(1, ys, yt);
%% Uncomment the following to remove libsvm-weight from your path, if you just added it
rmpath('D:\Matlab2020a\toolbox\libsvm-weights-3.24\matlab');

% %% Data normalization: Make the squared sum of each feature vector 1
% X=diag(sparse(1./sqrt(sum(X.^2,2))))*X;

%% Construct MMD matrix
e=cat(1,1/n*ones(n,1),1/m*ones(m,1));
M=e*e'*length(Cs);
for c=Cs'
    e=zeros(n+m,1);
    e(ys(1:n)==c)=1/sum(ys==c);
    e(n+find(yt==c))=-1/sum(yt==c);
    e(isinf(e))=0;
    M=M+e*e';
end
%M=M/norm(M,'fro');

%% Compute K; linear kernel; more complex kernels could be used
if nargin<7 || isempty(K)
    K=X* X';
end

%% Compute alpha
W=cat(1,Ws,wt*Wt); E=diag(cat(1,W,zeros(m-ml,1)));
alpha=((E+lambda*M)*K+sigma*speye(n+m,n+m))\(E*Y);

%% Classification
yt1Raw=K(n+ml+1:end,:)*alpha;
yt1=sign(yt1Raw); ids1=yt1==1;
yt1(~ids1)=Cs(1); yt1(ids1)=Cs(2);

%% Compute weight for the classifier
%f=norm(E*(Y-K*alpha))+trace(sigma*alpha'*K*alpha+alpha'*K*(lambda*M+gamma*L)*K*alpha);
yRaw=sign(K(1:n+ml,:)*alpha);
ids1=yRaw==1;
yRaw(~ids1)=Cs(1); yRaw(ids1)=Cs(2);
f=(yRaw==cat(1,ys, yt0))'*W/sum(W);


