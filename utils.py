
#bidsloader
import nibabel as nib
import numpy as np
from bidsio import BIDSLoader
import os

def bids_loader(bids_dir, output_bids_dir, gt_dir, gt_target_dir):
    file_list1 = os.listdir(bids_dir)
    file_list_nii1 = sorted([file for file in file_list1 if file.endswith(".nii.gz")])
    for i in range(len(file_list_nii1)):
        img1_path = bids_dir+"/"+file_list_nii1[i]
        img1 = nib.load(img1_path) #nifti 파일에서 이미지 영역만 가져오기
        img1_affine = img1.affine ########## img1 affine metrix 저장
        numpy1 = np.array(img1.dataobj) #numpy 자료형으로 변경
        numpy1 = numpy1.astype(np.int8)
        ##########Save as nifti###########
        img = nib.Nifti1Image(numpy1, img1_affine)  # Save axis for data (just identity)
        img.header.get_xyzt_units()
        fname = file_list_nii1[i].split(".")
        directory1 = "sub-r000s"+fname[0][-3:]
        directory2 = "ses-1"
        directory3 = "anat"
        path3 = f'{output_bids_dir}/'
        final_directory = path3+directory1+"/"+directory2+"/"+directory3+"/"
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        
        refname = directory1+"_"+directory2+"_space-MNI152NLin2009aSym_label_L_mask.nii.gz"
        img.to_filename(os.path.join(final_directory,'{}'.format(refname)))  # Save as NiBabel file
        # Automatically create dataset_description
    BIDSLoader.write_dataset_description(bids_root=path3, dataset_name = 'atlas2_prediction',author_names='postech')
    
    if not os.path.isdir(gt_target_dir):
        file_list1 = os.listdir(gt_dir)
        file_list_nii1 = sorted([file for file in file_list1 if file.endswith(".nii.gz")])

        for i in range(len(file_list_nii1)):
            img1_path = gt_dir+"/"+file_list_nii1[i]
            img1 = nib.load(img1_path) #nifti 파일에서 이미지 영역만 가져오기
            img1_affine = img1.affine ########## img1 affine metrix 저장
            numpy1 = np.array(img1.dataobj) #numpy 자료형으로 변경
            numpy1 = numpy1.astype(np.int8)
            ##########Save as nifti###########
            img = nib.Nifti1Image(numpy1, img1_affine)  # Save axis for data (just identity)
            img.header.get_xyzt_units()
            fname = file_list_nii1[i].split(".")
            directory1 = "sub-r000s"+fname[0][-3:]
            directory2 = "ses-1"
            directory3 = "anat"
            final_directory = gt_target_dir+directory1+"/"+directory2+"/"+directory3+"/"
            if not os.path.exists(final_directory):
                os.makedirs(final_directory)
            
            refname = directory1+"_"+directory2+"_space-MNI152NLin2009aSym_label_L_mask.nii.gz"
            img.to_filename(os.path.join(final_directory,'{}'.format(refname)))  # Save as NiBabel file
            # Automatically create dataset_description
        BIDSLoader.write_dataset_description(bids_root=gt_target_dir, dataset_name = 'atlas2_prediction',author_names='postech')
