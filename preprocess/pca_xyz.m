% compute PCA
% author: Liuhao Ge

clc;clear;close all;

data_dir='./';

subject_names={'P0','P1','P2','P3','P4','P5','P6','P7','P8'};
gesture_names={'1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y'};

JNT_NUM = 21;

for test_subject = 1:9
    display(test_subject)
    
    jnt_xyz=[];
    for sub_idx = 1:length(subject_names)
        for ges_idx = 1:length(gesture_names)
            gesture_dir = [data_dir subject_names{sub_idx} '/' gesture_names{ges_idx}];
            load([gesture_dir '/Volume_GT_XYZ.mat']);
            load([gesture_dir '/valid.mat']);
            tmp1 = permute(Volume_GT_XYZ, [1 3 2]);
            tmp2 = reshape(tmp1,[size(Volume_GT_XYZ,1),JNT_NUM*3]);
            if sub_idx~=test_subject
                jnt_xyz = [jnt_xyz; tmp2];
            end
        end
    end

    [PCA_coeff,score,PCA_latent_weight] = pca(jnt_xyz);
    PCA_mean_xyz = mean(jnt_xyz,1);

    save_dir = [data_dir subject_names{test_subject}];
    save([save_dir '/PCA_mean_xyz.mat'],'PCA_mean_xyz');
    save([save_dir '/PCA_coeff.mat'],'PCA_coeff');
    save([save_dir '/PCA_latent_weight.mat'],'PCA_latent_weight');
end