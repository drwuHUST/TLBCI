%% Study the effect of transfer learning in 3 different components of online cross-subject MI classification:
% 1. Data alignment
% 2. Spatial filtering
% 3. Classification
%
%% Compare 27 different algorithms:
% 1. CSP-LDA
% 2. CSP-CLDA
% 3. CSP-wAR
% 4. CCSP-LDA
% 5. CCSP-CLDA
% 6. CCSP-wAR
% 7. RCSP-LDA
% 8. RCSP-CLDA
% 9. RCSP-wAR
% 10. EA-CSP-LDA
% 11. EA-CSP-CLDA
% 12. EA-CSP-wAR
% 13. EA-CCSP-LDA
% 14. EA-CCSP-CLDA
% 15. EA-CCSP-wAR
% 16. EA-RCSP-LDA
% 17. EA-RCSP-CLDA
% 18. EA-RCSP-wAR
% 19. PS-CSP-LDA
% 20. PS-CSP-CLDA
% 21. PS-CSP-wAR
% 22. PS-CCSP-LDA
% 23. PS-CCSP-CLDA
% 24. PS-CCSP-wAR
% 25. PS-RCSP-LDA
% 26. PS-RCSP-CLDA
% 27. PS-RCSP-wAR
%
% We study online cross-subject transfer, by combining data from all other subjects to form a single source domain
% Experiment 1: on Dataset 1, 7 subjects
% Experiment 2: on Dataset 2a, using training data of the 9 subjects only
% Experiment 3: on Dataset 2a, using evaluation data of the 9 subjects only
%
% Need the covariance toolbox: https://github.com/alexandrebarachant/covariancetoolbox
%
%% Dongrui Wu, drwu@hust.edu.cn

clc; clearvars; close all; warning off all; rng('default');

nRepeat=30; % number of repeats to get statistically meaningful results
minN=4; % min number of target labeled samples
maxN=20; % max number of target labeled samples
nStep=4; % number of target labeled samples to add in each iteration
nAlgs=27; % Number of algorithms


for ds=1:2
    switch ds
        case 1
            files=dir('./Data1/A*.mat');
        case 2
            files=dir('./Data2/A*T.mat');
        case 3
            files=dir('./Data2/A*E.mat');
    end
    
    XRaw=[]; XallEA=[];  XallPS=[]; yAll=[]; nSubs=length(files);
    for s=1:nSubs
        s
        load([files(s).folder '\' files(s).name]);
        XRaw=cat(3,XRaw,X);  yAll=cat(1,yAll,y); nTrials=length(y);
        
        %% EA for all subjects
        % Need the covariance toolbox: https://github.com/alexandrebarachant/covariancetoolbox
        refEA=mean_covariances(covariances(X),'arithmetic');
        sqrtRefEA=refEA^(-1/2);
        XEA=nan(size(X));
        for j=1:length(y)
            XEA(:,:,j)=sqrtRefEA*X(:,:,j);
        end
        XallEA=cat(3,XallEA,XEA);
        
        %% PS for all subjects
        % Need the covariance toolbox: https://github.com/alexandrebarachant/covariancetoolbox
        refPS=mean_covariances(covariances(X),'riemann');
        sqrtRefPS=refPS^(-1/2);
        XPS=nan(size(X));
        for j=1:length(y)
            XPS(:,:,j)=sqrtRefPS*X(:,:,j);
        end
        XallPS=cat(3,XallPS,XPS);
    end
    labels=unique(y);
    
    %% Main loop
    Accs=cell(1,nSubs);
    for t=1:nSubs
        idsTarget=(t-1)*nTrials+1:t*nTrials;
        idsSource=1:nSubs*nTrials; idsSource(idsTarget)=[];
        XtRaw=XRaw(:,:,idsTarget);   yt=yAll(idsTarget);
        XsRaw=XRaw(:,:,idsSource); XsEA=XallEA(:,:,idsSource);
        XsPS=XallPS(:,:,idsSource);  ys=yAll(idsSource);
        ids1=find(ys==labels(1)); ids2=find(ys==labels(2));
        acc=zeros(nAlgs,nRepeat,floor((maxN-minN)/nStep)+1);
        
        parfor r=1:nRepeat
            [t,r]
            
            tempAcc=nan(nAlgs,floor((maxN-minN)/nStep)+1);
            idsTrain0=round(nTrials*rand)+(1:maxN);
            idsTrain0(idsTrain0>nTrials)=idsTrain0(idsTrain0>nTrials)-nTrials;
            while length(unique(yt(idsTrain0(1:minN))))==1 % make sure the first minN samples include two classes
                idsTrain0=round(nTrials*rand)+(1:maxN);
                idsTrain0(idsTrain0>nTrials)=idsTrain0(idsTrain0>nTrials)-nTrials;
            end
            idsTest0=1:nTrials; idsTest0(idsTrain0)=[];
            
            %% Online calibration
            for n=minN:nStep:maxN % select training trials from the training pool
                idsTrain=idsTrain0(1:n);
                idsTest=cat(2,idsTrain0(n+1:end),idsTest0);
                XtTestRaw=XtRaw(:,:,idsTest);
                ytTest=yt(idsTest);
                XtTrainRaw=XtRaw(:,:,idsTrain);
                ytTrain=yt(idsTrain);
                yTrain=cat(1,ytTrain,ys);
                
                %% %%% Case 1: CSP filters from the target subject only, no EA, no TLCSP
                % Target training data only
                % 1. CSP-LDA
                idxAlg=1;
                [fTrain,fTest]=CSPfeature(XtTrainRaw,ytTrain,XtTestRaw);
                LDA = fitcdiscr(fTrain,ytTrain);    yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 2. CSP-CLDA
                idxAlg=2;
                [fTrain,fTest]=CSPfeature(XtTrainRaw,ytTrain,cat(3,XsRaw,XtTestRaw));
                fTrain=cat(1,fTrain,fTest(1:length(ys),:)); fTest(1:length(ys),:)=[];
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 3. CSP-wAR
                idxAlg=3;
                yPredWAR3=OwAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR3);
                
                %% %%% Case 2: CCSP, use source data in CSP, no EA
                % 4. CCSP-LDA
                idxAlg=4;
                [fTrain,fTest]=CSPfeature(cat(3,XtTrainRaw,XsRaw),yTrain,XtTestRaw);
                LDA = fitcdiscr(fTrain(1:n,:),ytTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 5. CCSP-CLDA
                idxAlg=5;
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 6. CCSP+wAR
                idxAlg=6;
                yPredWAR6=OwAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR6);
                
                %% %%% Case 3: RCSP, use source data in CSP, no EA
                % 7. RCSP-LDA
                idxAlg=7;
                [fTrain,fTest]=RCSPfeature(XsRaw,ys,cat(3,XtTrainRaw,XtTestRaw),ytTrain);
                LDA = fitcdiscr(fTrain(1:n,:),ytTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 8. RCSP-CLDA
                idxAlg=8;
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 9. RCSP-wAR
                idxAlg=9;
                yPredWAR9=OwAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR9);
                
                %% %%%%%%%%%%%%%%% EA %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% EA for target data
                refEA=mean(covariances(XtTrainRaw),3); % EA reference matrix for new subject
                sqrtRefEA=refEA^(-1/2);
                XtEA=nan(size(XtRaw));
                for j=1:size(XtRaw,3)
                    XtEA(:,:,j)=sqrtRefEA*XtRaw(:,:,j);
                end
                
                %% %%% Case 4: EA + CSP, no TL in CSP
                % 10. EA-CSP-LDA
                idxAlg=10;
                [fTrain,fTest]=CSPfeature(XtEA(:,:,idsTrain),ytTrain,XtEA(:,:,idsTest));
                LDA = fitcdiscr(fTrain,ytTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 11. EA-CSP-CLDA
                idxAlg=11;
                [fTrain,fTest]=CSPfeature(XtEA(:,:,idsTrain),ytTrain,cat(3,XsEA,XtEA(:,:,idsTest)));
                fTrain=cat(1,fTrain,fTest(1:length(ys),:)); fTest(1:length(ys),:)=[];
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 12. EA-CSP-wAR
                idxAlg=12;
                yPredWAR12=OwAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR12);
                
                %% %%% Case 5: EA + CCSP
                % 13. EA-CCSP-LDA
                idxAlg=13;
                [fTrain,fTest]=CSPfeature(cat(3,XtEA(:,:,idsTrain),XsEA),yTrain,XtEA(:,:,idsTest));
                LDA = fitcdiscr(fTrain(1:n,:),ytTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 14. EA-CCSP-CLDA
                idxAlg=14;
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 15. EA-CCSP-wAR
                idxAlg=15;
                yPredWAR15=OwAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR15);
                
                %% %%% Case 6: EA + RCSP
                % 16. EA-RCSP-LDA
                idxAlg=16;
                [fTrain,fTest]=RCSPfeature(XsEA,ys,XtEA(:,:,[idsTrain idsTest]),ytTrain);
                LDA = fitcdiscr(fTrain(1:n,:),ytTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 17. EA-RCSP-CLDA
                idxAlg=17;
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 18. EA-RCSP-wAR
                idxAlg=18;
                yPredWAR18=OwAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR18);
                
                %% %%%%%%%%%%%%% PS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% PS for target data
                refPS=mean_covariances(covariances(XtTrainRaw),'riemann');
                sqrtRefPS=refPS^(-1/2);
                XtPS=nan(size(XtRaw));
                for j=1:size(XtRaw,3)
                    XtPS(:,:,j)=sqrtRefPS*XtRaw(:,:,j);
                end
                
                %% %%% Case 7: PS + CSP, no TL in CSP
                % 19. PS-CSP-LDA
                idxAlg=19;
                [fTrain,fTest]=CSPfeature(XtPS(:,:,idsTrain),ytTrain,cat(3,XsPS,XtPS(:,:,idsTest)));
                fTrain=cat(1,fTrain,fTest(1:length(ys),:)); fTest(1:length(ys),:)=[];
                LDA = fitcdiscr(fTrain(1:n,:),ytTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 20. PS-CSP-CLDA
                idxAlg=20;
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 21. PS-CSP-wAR
                idxAlg=21;
                yPredWAR21=OwAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR21);
                
                %% %%% Case 8: PS + CCSP
                % 22. PS-CCSP-LDA
                idxAlg=22;
                [fTrain,fTest]=CSPfeature(cat(3,XtPS(:,:,idsTrain),XsPS),yTrain,XtPS(:,:,idsTest));
                LDA = fitcdiscr(fTrain(1:n,:),ytTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 23. PS-CCSP-CLDA
                idxAlg=23;
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 24. PS-CCSP-wAR
                idxAlg=24;
                yPredWAR24=OwAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR24);
                
                %% %%% Case 9: PS + RCSP
                % 25. PS-RCSP-LDA
                idxAlg=25;
                [fTrain,fTest]=RCSPfeature(XsPS,ys,XtPS(:,:,[idsTrain idsTest]),ytTrain);
                LDA = fitcdiscr(fTrain(1:n,:),ytTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 26. PS-RCSP-CLDA
                idxAlg=26;
                LDA = fitcdiscr(fTrain,yTrain);  yPred=predict(LDA,fTest);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPred);
                
                % 27. PS-RCSP-wAR
                idxAlg=27;
                yPredWAR27=OwAR(fTrain(n+1:end,:),ys,cat(1,fTrain(1:n,:),fTest),ytTrain);
                tempAcc(idxAlg,(n-minN)/nStep+1)=100*mean(ytTest==yPredWAR27);
                
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
    save(['OnlineMI_dataset' num2str(ds) '.mat'],'Accs','nStep','minN','maxN','nAlgs','nSubs','mmAcc');
end