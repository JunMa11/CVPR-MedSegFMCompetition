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

from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient

# Taken from CVPR24 challenge code with change to np.unique
def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in np.unique(gt)[1:]: # skip bg
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)

# Taken from CVPR24 challenge code with change to np.unique
def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in np.unique(gt)[1:]: # skip bg
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return np.mean(nsd)

parser = argparse.ArgumentParser('Segmentation eavluation for docker containers', add_help=False)
parser.add_argument('-i', '--test_img_path', default='./3D_val_img', type=str, help='testing data path')
parser.add_argument('-val_gts','--validation_gts_path', default='./3D_val_gt', type=str, help='path to validation set (or final test set) GT files')
parser.add_argument('-o','--save_path', default='./seg', type=str, help='segmentation output path')
parser.add_argument('-d','--docker_folder_path', default='./team_docker', type=str, help='team docker path')
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

                # Calculate DSC and NSD
                # get spacing from the 
                img_npz = np.load(join(input_temp, case), allow_pickle=True)
                spacing = img_npz['spacing']
                instance_label = img_npz['text_prompts'].item()['instance_label']
                if instance_label == 0:
                    # note: the semantic labels may not be sequential
                    dsc = compute_multi_class_dsc(seg_npz, gt_npz)
                    nsd = compute_multi_class_nsd(seg_npz, gt_npz, spacing)
                elif instance_label == 1:
                    # make sure the instace labels are sequential: 0, 1, 2, 3, 4...
                    gt_npz = segmentation.relabel_sequential(gt_npz)[0]
                    dsc = compute_multi_class_dsc(seg_npz, gt_npz)
                    nsd = compute_multi_class_nsd(seg_npz, gt_npz, spacing)                    


                metric['DSC'].append(dsc)
                metric['NSD'].append(nsd)

                print(f"{case}: DSC={dsc:.4f}, NSD={nsd:.4f}")

            except Exception as e:
                print(f"Error processing {case}: {e}")
                metric['DSC'].append(0.0)
                metric['NSD'].append(0.0)

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
