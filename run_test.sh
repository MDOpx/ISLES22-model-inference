# ./run_test.sh ASAN_Task1_sample 0
ID=$1
INDEX=$2


if [ ${INDEX} -eq 1 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 1"
   ./test.sh ${ID} 3d_fullres_dwi_DWI
fi
if [ ${INDEX} -eq 2 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 2"
   ./test.sh ${ID} 3d_fullres_dwi_tf_t2f_DWI
fi
if [ ${INDEX} -eq 3 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 3"
   ./test.sh ${ID} 3d_fullres_dwi_mt_DWI
fi
if [ ${INDEX} -eq 4 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 4"
   ./test.sh ${ID} 3d_fullres_dwi_tf_mt_DWI
fi
if [ ${INDEX} -eq 5 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 5"
   ./test.sh ${ID} 3d_fullres_adc_dwi_ADC+DWI
fi
if [ ${INDEX} -eq 6 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 6"
   ./test.sh ${ID} 3d_fullres_adc_dwi_tf_t2_ADC+DWI
fi
if [ ${INDEX} -eq 7 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 7"
   ./test.sh ${ID} 3d_fullres_adc_dwi_tf_t1_ADC+DWI_4
fi
if [ ${INDEX} -eq 8 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 8"
   ./test.sh ${ID} 3d_fullres_adc_dwi_tf_t1_mt_ADC+DWI_4
fi
if [ ${INDEX} -eq 9 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 9"
   ./test.sh ${ID} 3d_fullres_adc_dwi_tf_t1_mt2_ADC+DWI_4
fi
if [ ${INDEX} -eq 10 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 10"
   ./test.sh ${ID} Ensemble_mean_3d_fullres_dwi_tf_mt_DWI_3d_fullres_adc_dwi_tf_t2_ADC+DWI
fi
if [ ${INDEX} -eq 11 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 11"
   ./test.sh ${ID} Ensemble_mean_3d_fullres_dwi_tf_mt_DWI_3d_fullres_adc_dwi_tf_t1_mt_ADC+DWI_4
fi
if [ ${INDEX} -eq 12 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 12"
   ./test.sh ${ID} Ensemble_mean_3d_fullres_dwi_tf_mt_DWI_3d_fullres_adc_dwi_tf_t1_mt2_ADC+DWI_4
fi
if [ ${INDEX} -eq 13 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 13"
   ./test.sh ${ID} Ensemble_max_3d_fullres_dwi_tf_mt_DWI_3d_fullres_adc_dwi_tf_t2_ADC+DWI
fi
if [ ${INDEX} -eq 14 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 14"
   ./test.sh ${ID} Ensemble_max_3d_fullres_dwi_tf_mt_DWI_3d_fullres_adc_dwi_tf_t1_mt_ADC+DWI_4
fi
if [ ${INDEX} -eq 15 -o ${INDEX} -eq 0 ];then
   sudo chmod -R 777 *
   echo "Start [$ID] Test 15"
   ./test.sh ${ID} Ensemble_max_3d_fullres_dwi_tf_mt_DWI_3d_fullres_adc_dwi_tf_t1_mt2_ADC+DWI_4
fi
