%% Study the effect of transfer learning in 4 different components of a closed-loop BCI system:
% 1. Data alignment
% 2. Signal processing
% 3. Feature engineering
% 4. Classification
%
%% Compare 13 different algorithms:
% 1. CSP-LDA
% 2. CSP-CLDA
% 3. CSP-wAR
% 4. CCSP-CLDA
% 5. CCSP-wAR
% 6. RCSP-CLDA
% 7. RCSP-wAR
% 8. EA-CSP-CLDA
% 9. EA-CSP-wAR
% 10. EA-CCSP-CLDA
% 11. EA-CCSP-wAR
% 12. EA-RCSP-CLDA
% 13. EA-RCSP-wAR
%
% We combine data from all other subjects to form a single source domain
%
%% Dongrui Wu, drwu@hust.edu.cn

clc; clearvars; close all; warning off all; rng('default');

nRepeat=30;
maxN=4;
minN=0;
nStep=4;
nAlgs=13;


for ds=1:2
    dataFolder=['./Data' num2str(ds) '/'];
    files=dir([dataFolder 'A*.mat']);
    XRaw=[]; XallEA=[];  yAll=[]; nSubs=length(files);
    for s=1:nSubs
        s
        load([dataFolder files(s).name]);
        XRaw=cat(3,XRaw,X);  yAll=cat(1,yAll,y); nTrials=length(y);
        %% EA for all subjects
        refEA=0; % reference matrix, Euclidean space
        for i=1:size(X,3)
            refEA=refEA+cov(X(:,:,i)');
        end
        refEA=refEA/size(X,3);
        sqrtRefEA=refEA^(-1/2);
        XEA=nan(size(X));
        for j=1:length(y)
            XEA(:,:,j)=sqrtRefEA*X(:,:,j);
        end
        XallEA=cat(3,XallEA,XEA);
    end
    labels=unique(y);
    
    %% Main loop
    Accs=cell(1,nSubs);
    for t=1:nSubs
        idsTarget=(t-1)*nTrials+1:t*nTrials;
        idsSource=1:nSubs*nTrials; idsSource(idsTarget)=[];
        XtRaw=XRaw(:,:,idsTarget); XtEA=XallEA(:,:,idsTarget);   yt=yAll(idsTarget);
        XsRaw=XRaw(:,:,idsSource); XsEA=XallEA(:,:,idsSource);   ys=yAll(idsSource);
        ids1=find(ys==labels(1)); ids2=find(ys==labels(2));
        acc=zeros(nAlgs,nRepeat,floor((maxN-minN)/nStep)+1);
        
        for r=1:nRepeat
            [t,r]
            
            tempAcc=nan(nAlgs,floor((maxN-minN)/nStep)+1);
            idsTrain0=datasample(1:nTrials,maxN,'replace',false);
            while length(unique(yt(idsTrain0(1:nStep))))==1
                idsTrain0=datasample(1:nTrials,maxN,'replace',false);
            end
            idsTest0=1:nTrials; idsTest0(idsTrain0)=[];
            
            %% Offline calibration
            for n=minN:nStep:maxN % select training trials from training pool
                idsTrain=idsTrain0(1:n);
                idsTest=cat(2,idsTrain0(n+1:end),idsTest0);
                XtTestRaw=XtRaw(:,:,idsTest);
                ytTest=yt(idsTest);
                XtTrainRaw=XtRaw(:,:,idsTrain);
                ytTrain=yt(idsTrain);
                
                %% %%% Case 1: CSP filters from the target subject only, no EA, no TLCSP
                % Target training data only
                if n>0
                    % 1. CSP-LDA
                    [fTrain,fTest]=CSPfeature(XtTrainRaw,ytTrain,XtTestRaw);
                    LDA = fitcdiscr(fTrain,ytTrain);    yPred=predict(LDA,fTest);
                    tempAcc(1,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                    
                    % 2. CSP-CLDA
                    [fTrain,fTest]=CSPfeature(XtTrainRaw,ytTrain,cat(3,XsRaw,XtTestRaw));
                    fTrain=cat(1,fTrain,fTest(1:length(ys),:)); fTest(1:length(ys),:)=[];
                    yTrain=cat(1,ytTrain,ys);
                    LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                    tempAcc(2,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                    
                    % 3. CSP-wAR
                    if n==minN || isnan(tempAcc(3,(n-minN)/nStep))
                        yPredWAR1=wAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain,[]);
                    else
                        yPredWAR1=wAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain,yPredWAR1(nStep+1:end));
                    end
                    tempAcc(3,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR1);
                end
                
                %% %%% Case 2: TLCSP, use source data in CSP
                % 4. CCSP-CLDA
                yTrain=cat(1,ytTrain,ys);
                [fTrain,fTest]=CSPfeature(cat(3,XtTrainRaw,XsRaw),yTrain,XtTestRaw);
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(4,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 5. CCSP+wAR
                if n==minN || isnan(tempAcc(5,(n-minN)/nStep))
                    yPredWAR2=wAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain,[]);
                else
                    yPredWAR2=wAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain,yPredWAR2(nStep+1:end));
                end
                tempAcc(5,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR2);
                
                %% %%% Case 3: RCSP, use source data in CSP
                % 6. RCSP-CLDA
                [fTrain,fTest]=RCSPfeature(XsRaw,ys,cat(3,XtTrainRaw,XtTestRaw),ytTrain);
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(6,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 7. RCSP-wAR
                if n==minN || isnan(tempAcc(7,(n-minN)/nStep))
                    yPredWAR3=wAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain,[]);
                else
                    yPredWAR3=wAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain,yPredWAR3(nStep+1:end));
                end
                tempAcc(7,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR3);
                
                %% %%% Case 4: EA + CSP
                if n>0
                    % 8. EA-CSP-CLDA
                    yTrain=cat(1,ytTrain,ys);
                    [fTrain,fTest]=CSPfeature(XtEA(:,:,idsTrain),ytTrain,cat(3,XsEA,XtEA(:,:,idsTest)));
                    fTrain=cat(1,fTrain,fTest(1:length(ys),:)); fTest(1:length(ys),:)=[];
                    LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                    tempAcc(8,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                    
                    % 9. EA-CSP-wAR
                    if n==minN || isnan(tempAcc(9,(n-minN)/nStep))
                        yPredWAR4=wAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain,[]);
                    else
                        yPredWAR4=wAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain,yPredWAR4(nStep+1:end));
                    end
                    tempAcc(9,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR4);
                end
                
                %% %%% Case 5: EA + TLCSP
                % 10. EA-CCSP-CLDA
                yTrain=cat(1,ytTrain,ys);
                [fTrain,fTest]=CSPfeature(cat(3,XtEA(:,:,idsTrain),XsEA),yTrain,XtEA(:,:,idsTest));
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(10,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 11. EA-CCSP-wAR
                if n==minN || isnan(tempAcc(12,(n-minN)/nStep))
                    yPredWAR5=wAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain,[]);
                else
                    yPredWAR5=wAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain,yPredWAR5(nStep+1:end));
                end
                tempAcc(11,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR5);
                
                %% %%% Case 6: EA + RCSP
                % 12. EA-RCSP-CLDA
                [fTrain,fTest]=RCSPfeature(XsEA,ys,XtEA(:,:,[idsTrain idsTest]),ytTrain);
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(12,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 13. EA-RCSP-wAR
                if n==minN || isnan(tempAcc(13,(n-minN)/nStep))
                    yPredWAR6=wAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain,[]);
                else
                    yPredWAR6=wAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain,yPredWAR6(nStep+1:end));
                end
                tempAcc(13,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR6);
                
            end
            acc(:,r,:)=tempAcc;
        end
        squeeze(mean(acc,2))
        Accs{t}=acc;
    end
    eval(['Accs' num2str(ds) '=Accs;']);
    mmAcc=zeros(nAlgs,(maxN-minN)/nStep+1);
    for t=1:nSubs
        mmAcc=mmAcc+squeeze(mean(Accs{t},2))/nSubs;
    end
    mmAcc
    save(['OfflineMIoverall_' num2str(ds) '.mat'],'Accs','nStep','minN','maxN','nAlgs','nSubs','mmAcc');
end