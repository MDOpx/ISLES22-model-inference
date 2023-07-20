# ./run.sh ASAN_Task1_sample 0
ID=$1
INDEX=$2

echo "Start [$ID] Test [$INDEX]"
if [ ${INDEX} -eq 1 -o ${INDEX} -eq 0 ];then
   cd nnunet_normal
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --model 3d_fullres_dwi --modal DWI
fi
if [ ${INDEX} -eq 2 -o ${INDEX} -eq 0 ];then
   cd nnunet_multitask
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --model 3d_fullres_dwi_tf_t2f --modal DWI
fi
if [ ${INDEX} -eq 3 -o ${INDEX} -eq 0 ];then
   cd nnunet_normal
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --model 3d_fullres_dwi_mt --modal DWI
fi
if [ ${INDEX} -eq 4 -o ${INDEX} -eq 0 ];then
   cd nnunet_normal
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --model 3d_fullres_dwi_tf_mt --modal DWI
fi
if [ ${INDEX} -eq 5 -o ${INDEX} -eq 0 ];then
   cd nnunet_normal
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --model 3d_fullres_adc_dwi --modal ADC+DWI
fi
if [ ${INDEX} -eq 6 -o ${INDEX} -eq 0 ];then
   cd nnunet_multitask
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --model 3d_fullres_adc_dwi_tf_t2 --modal ADC+DWI
fi
if [ ${INDEX} -eq 7 -o ${INDEX} -eq 0 ];then
   cd nnunet_multitask
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --model 3d_fullres_adc_dwi_tf_t1 3d_fullres_adc_dwi_tf_t1ce 3d_fullres_adc_dwi_tf_t2 3d_fullres_adc_dwi_tf_t2f --modal ADC+DWI
fi
if [ ${INDEX} -eq 8 -o ${INDEX} -eq 0 ];then
   cd nnunet_multitask
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --model 3d_fullres_adc_dwi_tf_t1_mt 3d_fullres_adc_dwi_tf_t1ce_mt 3d_fullres_adc_dwi_tf_t2_mt 3d_fullres_adc_dwi_tf_t2f_mt --modal ADC+DWI
fi
if [ ${INDEX} -eq 9 -o ${INDEX} -eq 0 ];then
   cd nnunet_multitask
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --model 3d_fullres_adc_dwi_tf_t1_mt2 3d_fullres_adc_dwi_tf_t1ce_mt2 3d_fullres_adc_dwi_tf_t2_mt2 3d_fullres_adc_dwi_tf_t2f_mt2 --modal ADC+DWI
fi
if [ ${INDEX} -eq 10 -o ${INDEX} -eq 0 ];then
   cd nnunet_normal
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --model 3d_fullres_dwi_tf_mt --modal DWI --model_ensemble_name 3d_fullres_adc_dwi_tf_t2_ADC+DWI
fi
if [ ${INDEX} -eq 11 -o ${INDEX} -eq 0 ];then
   cd nnunet_normal
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --model 3d_fullres_dwi_tf_mt --modal DWI --model_ensemble_name 3d_fullres_adc_dwi_tf_t1_mt_ADC+DWI_4 
fi
if [ ${INDEX} -eq 12 -o ${INDEX} -eq 0 ];then
   cd nnunet_normal
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --model 3d_fullres_dwi_tf_mt --modal DWI --model_ensemble_name 3d_fullres_adc_dwi_tf_t1_mt2_ADC+DWI_4
fi
if [ ${INDEX} -eq 13 -o ${INDEX} -eq 0 ];then
   cd nnunet_normal
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --modal_ensemble_method max --model 3d_fullres_dwi_tf_mt --modal DWI --model_ensemble_name 3d_fullres_adc_dwi_tf_t2_ADC+DWI
fi
if [ ${INDEX} -eq 14 -o ${INDEX} -eq 0 ];then
   cd nnunet_normal
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --modal_ensemble_method max --model 3d_fullres_dwi_tf_mt --modal DWI --model_ensemble_name 3d_fullres_adc_dwi_tf_t1_mt_ADC+DWI_4 
fi
if [ ${INDEX} -eq 15 -o ${INDEX} -eq 0 ];then
   cd nnunet_normal
   pip install -e.
   cd ..
   python nnUNet_external_test.py --taskid ${ID} --modal_ensemble_method max --model 3d_fullres_dwi_tf_mt --modal DWI --model_ensemble_name 3d_fullres_adc_dwi_tf_t1_mt2_ADC+DWI_4
fi
chmod -R 777 *
rm -r usr_folder