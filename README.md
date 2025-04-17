# CVPR-MedSegFMCompetition
 Foundation Models for Biomedical Image Segmentation

## Evaluation
The evaluation script `CVPR25_iter_eval.py` evaluates Docker submissions for the **CVPR25: Foundation Models for Interactive 3D Biomedical Image Segmentation Challenge** using an iterative refinement approach.

### Installation
Installation of packages for the evaluation script:
```
conda create -n cvpr_segfm_eval python=3.11 -y
conda activate cvpr_segfm_eval
pip install requirements.txt
```

Run the script as follows:

```bash
python CVPR25_iter_eval.py --docker_folder path/to/docker_submissions --test_img_path path/to/test_images --save_path path/to/output --verbose
```

### Arguments
- `--docker_folder` : Path to the directory containing submitted Docker containers (`.tar.gz`).
- `--test_img_path` : Path to the directory containing `.npz` test images.
- `--save_path` : Directory to save segmentation outputs and evaluation metrics.
- `--verbose` *(optional)* : Enables detailed output, including generated click coordinates.
- `--validation_gts_path` Path to validation / test set GT files. This is needed to prevent label leakage (val/test) during the challenge.

### Evaluation Process
1. **Loads Docker submissions** and processes test images one by one.
2. **Initial Prediction:** Uses a bounding box prompt to generate the first segmentation.
3. **Iterative Refinement:** Simulates up to 5 refinement clicks based on segmentation errors.
4. **Performance Metrics:** Computes **Dice Similarity Coefficient AUC (DSC_AUC), Normalized Surface Dice AUC (NSD_AUC), Final DSC, Final NSD, and Inference Time**.
5. **Outputs results** as `.npz` files and a CSV summary.

### Output
- Segmentation results are saved in the specified output directory. 
    -   Final prediction in the `segs` key
    -   All the 6 intermediate predictions in the `all_segs` key
- Metrics for each test case are compiled into a CSV file.

For more details, refer to the challenge page: https://www.codabench.org/competitions/5263/


### Clicks Accumulation in Image Input

During the prediction process, clicks are accumulated in the `clicks` key within the input `.npz` file.  

An example of a list stored in the `clicks` key for an image with 4 targets and after all 5 clicks:  

```json
[
    {"fg": [[46, 336, 343], [28, 233, 365], [28, 233, 365], [28, 233, 365]], "bg": [[28, 233, 366]]}, 
    {"fg": [[38, 210, 148]], "bg": [[6, 230, 284], [6, 230, 284], [6, 230, 284], [6, 230, 284]]}, 
    {"fg": [[12, 287, 262], [12, 287, 262], [12, 287, 262], [12, 287, 262], [12, 287, 262]], "bg": []}, 
    {"fg": [[28, 199, 180], [28, 199, 180], [28, 199, 180], [28, 199, 180], [28, 199, 180]], "bg": []}, 
]
```
### Clicks Order
We also provide the order in which the clicks were generated in a ancilliary key `clicks_order` that is a simple list with values `fg` and `bg`, e.g., `['fg', 'fg', 'bg']`, indicating that the first two clicks were foreground clicks and the last a background click. 
### Previous Prediction in Image Input

The input image also contains the `prev_pred` key which stores the prediction from the previous iteration. This is used only to help with submissions that are using the previous prediction as an additional input.

### No Bounding Box key
We also omit the `boxes` key in some of the validation and test samples as it is a bad prompt for some structures, such as vessels. In this case we simply skip the first inital prediction and only evaluate the models with 5 clicks using the same evaluation metrics.


### Upper Time Bound During Testing
We set a limit of 90 seconds per class during inference (whole docker run). If the inference time exceeds this bound, the corresponding DSC and NSD scores will be set as 0. When participants evaluate their models using the  `CVPR25_iter_eval.py` script they will receive a warning if their models exceed this limit.

There are two motivations for this setting
- The main focus of this competition is to prompt the interactive segmentation algorithm designs. Inference time should not be a huge concern/constraint for participants. 
- It is very hard to evaluate the real inference time within docker since implementations also affect the docker overhead.

### Final Script Output
The `CVPR25_iter_eval.py` script will produce the following outputs in the `--save_path` argument:
- `{teamname}_metrics.csv` that contains the following columns
    - `CaseName`: Test / Validation image filename
    - `TotalRunningTime`: Inference time taken for the image (all interactions)
    - `RunningTime_{i}`: Inference time for interactions [1-6], 1: bbox, 2-6: clicks
    - `DSC_AUC`: Area under DSC-to-Click curve metric
    - `NSD_AUC`: Area under NSD-to-Click curve metric
    - `DSC_Final`: DSC after final click
    - `NSD_Final`: NSD after final click
- `CASE_{i}.npz` - model output with keys:
    - `segs`: Final prediction for all classes
    - `all_segs`: All intermediate predictions of the model for interactions [1-6]

