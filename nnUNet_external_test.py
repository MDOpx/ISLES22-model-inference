import SimpleITK
import os
import glob
from utils import *
#nnunet
from nnunet.inference.predict import predict_from_folder
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.inference.ensemble_predictions import merge
# Preprocessing Code
import shutil
import argparse
#   python nnUNet_external_test.py --taskid ASAN_Task1 --model 3d_fullres_dwi_tf_mt --modal DWI --model_ensemble_name 3d_fullres_adc_dwi_tf_t2_ADC+DWI

parser = argparse.ArgumentParser(description='External_nnUNet')
parser.add_argument('--taskid', default = 'ASAN_Task1', required=True, help='external_task_id')
parser.add_argument('--model', nargs='+', default = '3d_fullres_dwi_tf_t2f', required=True, help='pretrained_nnunet_model_name')
parser.add_argument('--model_ensemble_name', required=False, help='target model_ensemble_name')
parser.add_argument('--modal', default = 'DWI', required=True, help='modal_type (e.g. ADC, DWI, ADC+DWI)')
parser.add_argument('--model_ensemble_method', default = 'mean', required=False) #mean max
opt = parser.parse_args()

ID = opt.taskid
MODEL_list = opt.model
MODAL = opt.modal
MODEL_ENSEMBLE = opt.model_ensemble_name 
MODEL_ENSEMBLE_METHOD = opt.model_ensemble_method
rtpath = os.getcwd()
input_dir = f'{rtpath}/external_dataset/{ID}/imagesTs'
if len(MODEL_list) == 1:
    output_bids_dir = f'{rtpath}/results_bids/{ID}_{MODEL_list[0]}_{MODAL}'
    if not os.path.isdir(output_bids_dir): os.makedirs(output_bids_dir)
    output_dir = f'{rtpath}/results/{ID}_{MODEL_list[0]}_{MODAL}'
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
else:
    output_bids_dir = f'{rtpath}/results_bids/{ID}_{MODEL_list[0]}_{MODAL}_{len(MODEL_list)}'
    if not os.path.isdir(output_bids_dir): os.makedirs(output_bids_dir)
    output_dir = f'{rtpath}/results/{ID}_{MODEL_list[0]}_{MODAL}_{len(MODEL_list)}'
    if not os.path.isdir(output_dir): os.makedirs(output_dir)

input1_list = sorted(glob.glob(f'{input_dir}/*_0000.nii.gz')) #adc
input2_list = sorted(glob.glob(f'{input_dir}/*_0001.nii.gz')) #dwi

# ## Step0. Save in User Folder
print('\nStep0. Save in User Folder\n')
sdir = f'{rtpath}/usr_folder'

if os.path.exists(sdir): shutil.rmtree(sdir)
if not os.path.exists(sdir): 
    os.makedirs(sdir)
    step1_dir1 = f'{rtpath}/usr_folder/step1_adc'
    if not os.path.exists(step1_dir1):os.makedirs(step1_dir1)
    step1_dir2 = f'{rtpath}/usr_folder/step1_dwi'
    if not os.path.exists(step1_dir2):os.makedirs(step1_dir2)
    if MODAL != 'ADC+DWI':
        for idx in range(len(input1_list)):
            id = input1_list[idx].split('/')[-1][:-12] #Consider Index Number With New Dataset Name Format
            #print(id)
            shutil.copyfile(input1_list[idx], f'{step1_dir1}/{id}_0000.nii.gz') 
            shutil.copyfile(input2_list[idx], f'{step1_dir2}/{id}_0000.nii.gz') 

## Step2. nnUNet
print('\nStep2. nnUNet\n')

if MODEL_ENSEMBLE != None:
    results_list = glob.glob(f'{rtpath}/results/{ID}_{MODEL_ENSEMBLE}/results_*')
    if len(results_list) == 0:
        target_pkl = f'{rtpath}/results/{ID}_{MODEL_ENSEMBLE}'
    else:
        target_pkl = f'{rtpath}/results/{ID}_{MODEL_ENSEMBLE}/results_0'
else:
    target_pkl = None;

# 3d Network Run
model_folder_name = []
if len(MODEL_list) == 1:
    model_folder_name.append(f'{rtpath}/nnUNet_model/{MODEL_list[0]}')
else:
    for idx in range(len(opt.model)):
        model_folder_name.append(f'{rtpath}/nnUNet_model/{MODEL_list[idx]}')

if len(MODEL_list) == 1:
    if target_pkl != None:
        output_dir = f'{rtpath}/results/{ID}_{MODEL_list[0]}_{MODAL}_tmp'
        if not os.path.isdir(output_dir): os.makedirs(output_dir)

    if MODAL == 'ADC+DWI':
        predict_from_folder(model_folder_name[0], input_dir, output_dir, None, True, 6, 
                            2, None, 0, 1, True,overwrite_existing=True, mode="normal", overwrite_all_in_gpu=None,
                            mixed_precision=True,step_size=0.5, checkpoint_name='model_final_checkpoint',pkl_folder = target_pkl)
    elif MODAL == 'ADC':
        predict_from_folder(model_folder_name[0], step1_dir1, output_dir, None, True, 6, 
                            2, None, 0, 1, True,overwrite_existing=True, mode="normal", overwrite_all_in_gpu=None,
                            mixed_precision=True,step_size=0.5, checkpoint_name='model_final_checkpoint',pkl_folder = target_pkl)
    elif MODAL == 'DWI':
        predict_from_folder(model_folder_name[0], step1_dir2, output_dir, None, True, 6, 
                            2, None, 0, 1, True,overwrite_existing=True, mode="normal", overwrite_all_in_gpu=None,
                            mixed_precision=True,step_size=0.5, checkpoint_name='model_final_checkpoint',pkl_folder = target_pkl)
else:
    step2 = []
    for idx in range(len(MODEL_list)):
        step2_tmp = f'{output_dir}/results_{idx}'
        if not os.path.exists(step2_tmp):os.makedirs(step2_tmp)
        step2.append(step2_tmp)

    for idx in range(len(MODEL_list)):
        if MODAL == 'ADC+DWI':
            predict_from_folder(model_folder_name[idx], input_dir, step2[idx], None, True, 6, 
                                2, None, 0, 1, True,overwrite_existing=True, mode="normal", overwrite_all_in_gpu=None,
                                mixed_precision=True,step_size=0.5, checkpoint_name='model_final_checkpoint',pkl_folder = target_pkl)
        elif MODAL == 'ADC':
            predict_from_folder(model_folder_name[idx], step1_dir1, step2[idx], None, True, 6, 
                                2, None, 0, 1, True,overwrite_existing=True, mode="normal", overwrite_all_in_gpu=None,
                                mixed_precision=True,step_size=0.5, checkpoint_name='model_final_checkpoint',pkl_folder = target_pkl)
        elif MODAL == 'DWI':
            predict_from_folder(model_folder_name[idx], step1_dir2, step2[idx], None, True, 6, 
                                2, None, 0, 1, True,overwrite_existing=True, mode="normal", overwrite_all_in_gpu=None,
                                mixed_precision=True,step_size=0.5, checkpoint_name='model_final_checkpoint',pkl_folder = target_pkl)
    # Step 3_2 Ensemble
    merge(step2,output_dir,2,True,None)

if target_pkl == None:
    bids_loader(output_dir, output_bids_dir, f'{rtpath}/external_dataset/{ID}/labelsTs', f'{rtpath}/results_bids/{ID}_GroundTruth/')
        
# Step 4 Ensemble
if MODEL_ENSEMBLE != None:
    output_bids_dir = f'{rtpath}/results_bids/{ID}_Ensemble_{MODEL_ENSEMBLE_METHOD}_{MODEL_list[0]}_{MODAL}_{MODEL_ENSEMBLE}'
    if not os.path.isdir(output_bids_dir): os.makedirs(output_bids_dir)
    output_dir = f'{rtpath}/results/{ID}_Ensemble_{MODEL_ENSEMBLE_METHOD}_{MODEL_list[0]}_{MODAL}_{MODEL_ENSEMBLE}'
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    
    if MODEL_ENSEMBLE_METHOD == 'mean':
        print('mode:merge')
        results_list = glob.glob(f'{rtpath}/results/{ID}_{MODEL_ENSEMBLE}/results_*')
        merge_dir = []
        if len(results_list) == 0:
            merge_dir.append(f'{rtpath}/results/{ID}_{MODEL_ENSEMBLE}')
        else:
            for results_dir in results_list:
                merge_dir.append(results_dir)
        merge_dir.append(f'{rtpath}/results/{ID}_{MODEL_list[0]}_{MODAL}_tmp')
        merge(merge_dir,output_dir,2,True,None)
    else:
        print('mode:union')
        merge_dir1 = sorted(glob.glob(f'{rtpath}/results/{ID}_{MODEL_ENSEMBLE}/*.nii.gz'))
        merge_dir2 = sorted(glob.glob(f'{rtpath}/results/{ID}_{MODEL_list[0]}_{MODAL}_tmp/*.nii.gz'))
        for idx in range(len(merge_dir1)):
            id = merge_dir1[idx].split('/')[-1]
            prediction_1 = SimpleITK.ReadImage(merge_dir1[idx])
            prediction_2 = SimpleITK.ReadImage(merge_dir2[idx])
            origin, spacing, direction = prediction_1.GetOrigin(), prediction_1.GetSpacing(), prediction_1.GetDirection()
            union_ad = np.logical_or(SimpleITK.GetArrayFromImage(prediction_1), SimpleITK.GetArrayFromImage(prediction_2)) #union
            union_ad = union_ad.astype(np.int8)
            prediction = np.squeeze(union_ad).astype(int)

            output_image = SimpleITK.GetImageFromArray(prediction)
            output_image.SetOrigin(origin), output_image.SetSpacing(spacing), output_image.SetDirection(direction)
        
            SimpleITK.WriteImage(output_image, f'{output_dir}/{id}')
    shutil.rmtree(f'{rtpath}/results/{ID}_{MODEL_list[0]}_{MODAL}_tmp')
    bids_loader(output_dir, output_bids_dir, f'{rtpath}/external_dataset/{ID}/labelsTs', f'{rtpath}/results_bids/{ID}_GroundTruth/')
            
