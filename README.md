# CVPR-MedSegFMCompetition
 Foundation Models for Biomedical Image Segmentation

## Evaluation
The evaluation script `CVPR25_iter_eval.py` evaluates Docker submissions for the **CVPR25 Segment Anything in Medical Images on a Laptop Challenge** using an iterative refinement approach.


Run the script as follows:

```bash
python CVPR25_iter_eval.py --docker_folder path/to/docker_submissions --test_img_path path/to/test_images --save_path path/to/output --verbose
```

### Arguments
- `--docker_folder` : Path to the directory containing submitted Docker containers (`.tar.gz`).
- `--test_img_path` : Path to the directory containing `.npz` test images.
- `--save_path` : Directory to save segmentation outputs and evaluation metrics.
- `--verbose` *(optional)* : Enables detailed output, including generated click coordinates.

### Evaluation Process
1. **Loads Docker submissions** and processes test images one by one.
2. **Initial Prediction:** Uses a bounding box prompt to generate the first segmentation.
3. **Iterative Refinement:** Simulates up to 5 refinement clicks based on segmentation errors.
4. **Performance Metrics:** Computes **Dice Similarity Coefficient AUC (DSC_AUC), Normalized Surface Dice AUC (NSD_AUC), Final DSC, Final NSD, and Inference Time**.
5. **Outputs results** as `.npz` files and a CSV summary.

### Output
- Segmentation results are saved in the specified output directory.
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
