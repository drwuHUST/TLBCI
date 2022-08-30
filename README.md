# TLBCI
Matlab source code of the paper (2020_TLBCI_INS_Wu_v3.pdf in this folder, or http://arxiv.org/abs/2007.03746):

D. Wu, X. Jiang, R. Peng, Transfer Learning for Motor Imagery Based Brain-Computer Interfaces: A Tutorial, Neural Networks, 2022, submitted.

Run main_Preprocess.m to preprocess the raw EEG data downloaded from BCI Competition IV.

Run main_MI_offline_crossSubject.m and main_MI_online_crossSubject.m to obtain the results in the paper.

Note: svmtrain in wAR.m uses the weighted libsvm at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#weights_for_data_instances, instead of the traditional (unweighted) svm. Please make sure weighted libsvm is compiled and installed correctly (the .mex files are in the Matlab path) before running the Matlab code.
