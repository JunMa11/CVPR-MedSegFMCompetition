"""
The code was adapted from the MICCAI FLARE Challenge
https://github.com/JunMa11/FLARE

The testing images will be evaluated one by one.

Folder structure:
CVPR25_text_eval.py
- team_docker
    - teamname.tar.gz # submitted docker containers from participants
- test_demo
    - imgs
        - case1.npz  # testing image
        - case2.npz  
        - ...   
- demo_seg  # segmentation results *******segmentation key: ['segs']*******
    - case1.npz  # segmentation file name is the same as the testing image name
    - case2.npz  
    - ...
"""

import os
join = os.path.join
import shutil
import time
import torch
import argparse
from collections import OrderedDict
import pandas as pd
import numpy as np
from skimage import segmentation
from scipy.optimize import linear_sum_assignment
import cc3d
import SimpleITK as sitk

from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient

def compute_multi_class_dsc(gt, seg, label_ids):
    dsc = [None] * len(label_ids)
    for idx, i in enumerate(label_ids):
        gt_i = gt == i
        seg_i = seg == i
        dsc[idx] = compute_dice_coefficient(gt_i, seg_i)

    return np.nanmean(dsc)

def compute_multi_class_nsd(gt, seg, spacing, label_ids, tolerance=2.0):
    nsd = []
    nsd = [None] * len(label_ids)
    for idx, i in enumerate(label_ids):
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(gt_i, seg_i, spacing_mm=spacing)
        nsd[idx] = compute_surface_dice_at_tolerance(surface_distance, tolerance)
    return np.nanmean(nsd)

def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    x = x.ravel()
    y = y.ravel()
    
    # preallocate a 'contact map' matrix
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    
    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image 
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp

def eval_tp_fp_fn(masks_true, masks_pred, threshold=0.5):
    num_inst_gt = np.max(masks_true)
    num_inst_seg = np.max(masks_pred)
    if num_inst_seg>0:
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
            # for k,th in enumerate(threshold):
        tp = _true_positive(iou, threshold)
        fp = num_inst_seg - tp
        fn = num_inst_gt - tp
    else:
        # print('No segmentation results!')
        tp = 0
        fp = 0
        fn = 0
        
    return tp, fp, fn

parser = argparse.ArgumentParser('Segmentation eavluation for docker containers', add_help=False)
parser.add_argument('-i', '--test_img_path', default='./3D_val_npz', type=str, help='testing data path')
parser.add_argument('-val_gts','--validation_gts_path', default='./3D_val_gt_text_seg', type=str, help='path to validation set (or final test set) GT files')
parser.add_argument('-o','--save_path', default='./outputs', type=str, help='segmentation output path')
parser.add_argument('-d','--docker_folder_path', default='./team_dockers', type=str, help='team docker path')
args = parser.parse_args()  

test_img_path = args.test_img_path
validation_gts_path = args.validation_gts_path
save_path = args.save_path
docker_path = args.docker_folder_path

input_temp = './inputs/'
output_temp = './outputs'
os.makedirs(save_path, exist_ok=True)

dockers = sorted(os.listdir(docker_path))
test_cases = sorted(os.listdir(test_img_path))

for docker in dockers:
    try:
        # create temp folers for inference one-by-one
        if os.path.exists(input_temp):
            shutil.rmtree(input_temp)
        if os.path.exists(output_temp):
            shutil.rmtree(output_temp)
        os.makedirs(input_temp)
        os.makedirs(output_temp)

        # load docker and create a new folder to save segmentation results
        teamname = docker.split('.')[0].lower()
        print('teamname docker: ', docker)
        os.system('docker image load -i {}'.format(join(docker_path, docker)))

        # create a new folder to save segmentation results
        team_outpath = join(save_path, teamname)
        if os.path.exists(team_outpath):
            shutil.rmtree(team_outpath)
        os.mkdir(team_outpath)
        os.system('chmod -R 777 ./* ')  # give permission to all files

        # initialize the metric dictionary
        metric = OrderedDict()
        metric['CaseName'] = []
        metric['RunningTime'] = []  
        metric['DSC'] = []
        metric['NSD'] = []
        metric['F1'] = []

        # To obtain the running time for each case, testing cases are inferred one-by-one
        for case in test_cases:
            shutil.copy(join(test_img_path, case), input_temp)
            cmd = 'docker container run --gpus "device=0" -m 32G --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(teamname, teamname)
            print(teamname, ' docker command:', cmd, '\n', 'testing image name:', case)

            # run the docker container and measure inference time
            start_time = time.time()
            try:
                os.system(cmd)
            except Exception as e:
                print('inference error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(case, e)
            real_running_time = time.time() - start_time
            print(f"{case} finished! Inference time: {real_running_time}")

            # save metrics
            metric['CaseName'].append(case)
            metric['RunningTime'].append(real_running_time)

            # Metric calculation (DSC and NSD)
            seg_name = case
            gt_path = join(validation_gts_path, seg_name)
            seg_path = join(output_temp, seg_name)

            try:
                # Load ground truth and segmentation masks
                gt_npz = np.load(gt_path, allow_pickle=True)['gts']
                seg_npz = np.load(seg_path, allow_pickle=True)['segs']

                gt_npz = gt_npz.astype(np.uint8)
                seg_npz = seg_npz.astype(np.uint8)

                # Calculate DSC and NSD
                img_npz = np.load(join(input_temp, case), allow_pickle=True)
                spacing = img_npz['spacing']
                instance_label = img_npz['text_prompts'].item()['instance_label']

                class_ids = sorted([int(k) for k in img_npz['text_prompts'].item() if k != "instance_label"])
                class_ids_array = np.array(class_ids, dtype=np.int32)

                if instance_label == 0:     # semantic masks
                    # note: the semantic labels may not be sequential
                    dsc = compute_multi_class_dsc(gt_npz, seg_npz, class_ids_array)
                    nsd = compute_multi_class_nsd(gt_npz, seg_npz, spacing, class_ids_array)
                    f1_score = np.NaN
                elif instance_label == 1:  # instance masks
                    # Calculate F1 instead
                    if len(np.unique(seg_npz)) == 2:
                        print("converting segmentation to instance masks")
                        # convert prediction masks from binary to instance
                        tumor_inst, tumor_n = cc3d.connected_components(seg_npz, connectivity=6, return_N=True)

                        # put the tumor instances back to gt_data_ori
                        seg_npz[tumor_inst > 0] = (tumor_inst[tumor_inst > 0] + np.max(seg_npz))

                    gt_npz = segmentation.relabel_sequential(gt_npz)[0]
                    seg_npz = segmentation.relabel_sequential(seg_npz)[0]

                    tp, fp, fn = eval_tp_fp_fn(gt_npz, seg_npz)        # default f1 overlap threshold is 0.5
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    # Set DSC and NSD to None for instance masks
                    dsc = None
                    nsd = None
                                     
                metric['DSC'].append(round(dsc, 4) if dsc is not None else np.NAN)
                metric['NSD'].append(round(nsd, 4) if nsd is not None else np.NAN)
                metric['F1'].append(round(f1_score, 4) if f1_score is not None else np.NAN)

                print(f"{case}: DSC={dsc if dsc is not None else np.NAN}, NSD={nsd if nsd is not None else np.NAN}, F1={f1_score}")
            
            except Exception as e:
                print(f"Error processing {case}: {e}")

            # the segmentation file name should be the same as the testing image name
            try:
                os.rename(join(output_temp, seg_name), join(team_outpath, seg_name))
            except:
                print(f"{join(output_temp, seg_name)}, {join(team_outpath, seg_name)}")
                print("Wrong segmentation name!!! It should be the same as image_name")

            os.remove(join(input_temp, case))   # Moves the segmentation output file from output_temp to the appropriate team folder in demo_seg.

        # save the metrics to a CSV file
        metric_df = pd.DataFrame(metric)
        metric_df.to_csv(join(team_outpath, teamname + '_metrics.csv'), index=False)
        print(f"Metrics saved to {join(team_outpath, teamname + '_metrics.csv')}")

        # clean up
        torch.cuda.empty_cache()
        os.system("docker rmi {}:latest".format(teamname))
        shutil.rmtree(input_temp)
        shutil.rmtree(output_temp)

    except Exception as e:
        print(e)
