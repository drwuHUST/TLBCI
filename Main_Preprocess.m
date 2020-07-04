%% Preprocessing the EEG data downloaded from BCI Competition IV.
%% Dongrui Wu, drwu@hust.edu.cn


clc; clearvars;

% % --------------------- Dataset 1: MI Data 1------------------------------
% Dataset 1: http://www.bbci.de/competition/iv/desc_1.html
% 7 subjects, 100 trials in each class, 59 EEG channels

dataFolder='D:\Data\BCICIV_1\';
files=dir([dataFolder 'BCICIV_ca*.mat']);
ref=[];
for s=1:length(files)
    s
    load([dataFolder files(s).name]); fs=nfo.fs;
    EEG=.1*double(cnt);
    b=fir1(50,2*[8 30]/nfo.fs);%FIRfiltDesign(nfo.fs,8,30,[],[],1)
    EEG=filter(b,1,EEG);
    y=mrk.y'; %(-1 for class one or 1 for class two)
    nTrials=length(y);
    X=nan(size(EEG,2),300,nTrials);
    for i=1:nTrials
        X(:,:,i)=EEG(mrk.pos(i)+0.5*nfo.fs:mrk.pos(i)+3.5*nfo.fs-1,:)'; % [0.5-3.5] seconds epoch, channels*Times
    end
    save(['./Data1/A' num2str(s) '.mat'],'X','y','fs');
end


%% --------------------- Dataset 2a: MI Data 2a ------------------------------
% Dataset 2a: http://www.bbci.de/competition/iv/desc_2a.pdf
% 9 subjects, 72 trials in each class, 22 EEG channels
ref=[];
dataFolder='D:\Data\BCICIV_2a\';
files=dir([dataFolder '*T.gdf']);
for s=1:length(files)
    s
    try
        [EEG, h] = sload([dataFolder files(s).name]); % need to enable bioSig toolbox
    catch
        run('D:\Matlab2020a\toolbox\biosig4octmat-3.3.0\biosig_installer.m');
        [EEG, h] = sload([dataFolder files(s).name]); % need to enable bioSig toolbox
    end
    EEG(:,end-2:end)=[]; % last three channels are EOG
    for i=1:size(EEG,2)
        EEG(isnan(EEG(:,i)),i)=nanmean(EEG(:,i));
    end
    b=fir1(50,2*[8 30]/h.SampleRate); %FIRfiltDesign(h.SampleRate,8,30,22,[],1);
    EEG=filter(b,1,EEG); %band pass filetring
    ids1=h.EVENT.POS(h.EVENT.TYP==769); % left hand
    ids2=h.EVENT.POS(h.EVENT.TYP==770); % right hand
    y=[-ones(length(ids1),1); ones(length(ids2),1)];
    ids=[ids1; ids2];
    X=[];
    for i=length(ids):-1:1
        X(:,:,i)=EEG(ids(i)+round(.5*h.SampleRate):ids(i)+round(3.5*h.SampleRate-1),:)';
    end
    [~,index]=sort(ids);
    y=y(index); X=X(:,:,index);
    nTrials=length(y);
    fs=h.SampleRate;
    save(['./Data2/A' num2str(s) '.mat'],'X','y','fs');
end

