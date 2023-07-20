from bidsio import BIDSLoader
import pandas as pd
import json
from collections import defaultdict
import numpy as np
from multiprocessing import Pool
from settings import eval_settings
from glob import glob
import openpyxl


def evaluate(bids_loader: BIDSLoader,
             scoring_functions: dict) -> dict:
    '''
    Evaluates the prediction:truth pairs stored in the loader according to the scoring functions. Returns a dict
    containing the scores for each pair, keyed identically to scoring_functions.
    Parameters
    ----------
    bids_loader : BIDSLoader
        BIDSLoader containing predictions in .data_list and the ground_truth in .target_list.
    scoring_functions : dict
        Dictionary of scoring functions to use to evaluate predictions, keyed by the desired output name.
    Returns
    -------
    dict [list]
        Dictionary containing lists of scores for every sample. The intended use is to gather all dicts and pass them
        to the aggregator before writing out results.
    '''
    score_results = defaultdict(list)
    # Iterate through data
    for prediction, truth in bids_loader.load_batches():
        # Score
        for score_name, score in scoring_functions.items():
            scores = score(truth=truth, prediction=prediction, batchwise=True)
            score_results[score_name] += scores
    #print(score_results)
    return score_results


def aggregate_scores(scores, aggregates):
    '''
    Returns the aggregate measures in scores.
    Parameters
    ----------
    scores : dict
        Dictionaries of score_name:sample_scores to aggregate.
    aggregates : list
        List of strings containing valid keys of pandas.Series.describe().

    Returns
    -------

    '''
    score_des = pd.DataFrame(scores).describe()
    score_summary = defaultdict(dict)
    for score_name in score_des.keys():
        for score_agg in score_des[score_name].keys():
            if(score_agg in aggregates):
                score_summary[score_name][score_agg] = score_des[score_name][score_agg]
    return score_summary


def merge_dict(list_of_dicts: list) -> defaultdict:
    '''
    Merges the dicts of the list into a single dict.
    Parameters
    ----------
    list_of_dicts : list
        List of dicts to combine.

    Returns
    -------
    defaultdict
    '''
    merged_dict = defaultdict(list)
    for d in list_of_dicts:
        for key, value in d.items():
            merged_dict[key] += value
    return merged_dict


def make_subloader_and_evaluate(data_list, data_shape, target_list, target_shape, eval_settings):
    # We can't pass the BIDSLoader object directly, so we have to recreate it here. The sample BIDS dataset allows
    # us to quickly initialize a BIDSLoader since the dataset is small.
    local_loader = BIDSLoader(data_root=[eval_settings["SampleBIDS"]],
                              target_root=[eval_settings["SampleBIDS"]],
                              data_derivatives_names=['atlas2'],
                              target_derivatives_names=['atlas2'],
                              target_entities=[{"suffix": "T1w"}],
                              data_entities=[{"subject": "001"}])

    # Copy relevant properties into this worker's BIDSLoader
    local_loader.data_list = data_list
    local_loader.target_list = target_list
    local_loader.batch_size = eval_settings['LoaderBatchSize']
    local_loader.data_shape = data_shape  # Needed since sample image is 1x1x1
    local_loader.target_shape = target_shape  # Needed since sample image is 1x1x1
    #print(data_list)
    #asdasd
    return evaluate(local_loader, eval_settings['ScoringFunctions'])


if __name__ == "__main__":
    BIDSLoader.write_dataset_description(bids_root=eval_settings['PredictionRoot'],dataset_name='atlas2_prediction',author_names='postech')
    BIDSLoader.write_dataset_description(bids_root=eval_settings['GroundTruthRoot'],dataset_name='atlas2',author_names='postech')

    # Get data to pass to workers
    loader = BIDSLoader(data_root=[eval_settings['PredictionRoot']],
                        target_root=[eval_settings['GroundTruthRoot']],
                        data_derivatives_names=eval_settings['PredictionBIDSDerivativeName'],
                        target_derivatives_names=eval_settings['GroundTruthBIDSDerivativeName'],
                        target_entities=[eval_settings['GroundTruthEntities']],
                        data_entities=[eval_settings['PredictionEntities']])
    # Parallelize data
    num_proc = eval_settings['Multiprocessing']
    loader_idx_list = np.floor(np.linspace(0, len(loader), num_proc+1)).astype(int)
    pool_arg_list = []
    # Break up dataset into roughly equal portions
    for idx in range(num_proc):
        start_idx, end_idx = loader_idx_list[idx:idx+2]
        data_list = loader.data_list[start_idx:end_idx]
        target_list = loader.target_list[start_idx:end_idx]
        pool_arg_list.append([data_list, loader.data_shape, target_list, loader.target_shape, eval_settings])

    # Parallel stuff
    pool = Pool(num_proc)
    pool_scores = pool.starmap(make_subloader_and_evaluate, pool_arg_list)

    rtpath = eval_settings['PredictionRoot']
    input_list = sorted(glob(f'{rtpath}/sub*'))
    Dice_list = []
    VolumeDiff_list = []
    LesionCount_list = []
    F1_list = []
    for ix in range(len(pool_scores)):
        for ix2 in range(len(pool_scores[ix]['Dice'])):
            Dice_list.append(pool_scores[ix]['Dice'][ix2])
            F1_list.append(pool_scores[ix]['Lesionwise F1-Score'][ix2])
            VolumeDiff_list.append(pool_scores[ix]['Volume Difference'][ix2])
            LesionCount_list.append(pool_scores[ix]['Simple Lesion Count'][ix2])
    # Combine scores into single dict
    scores_dict = merge_dict(pool_scores)

    # Aggregate scores together
    score_summary = aggregate_scores(scores_dict, eval_settings["Aggregates"])

    # Write out
    
    wb = openpyxl.Workbook()
    sheet1 = wb['Sheet']
    sheet1.title = 'Evaluation'
    sheet1['A1'] = 'id'
    sheet1['B1'] = 'Dice'
    sheet1['C1'] = 'F1'
    sheet1['D1'] = 'VolumeDiff'
    sheet1['E1'] = 'LesionCount'
    # print(len(input_list))
    # print(len(Dice_list))
    # print(len(F1_list))
    # print(len(VolumeDiff_list))
    # print(len(LesionCount_list))
    for idx in range(len(input_list)):
        sheet1.append([input_list[idx], Dice_list[idx], F1_list[idx], VolumeDiff_list[idx], LesionCount_list[idx]])

    wb.save(eval_settings['MetricsOutputPath'])  

    # f = open(eval_settings['MetricsOutputPath'], 'w')
    # json.dump(score_summary, f)
    # json.dump(score_summary, f)
    # f.close()
