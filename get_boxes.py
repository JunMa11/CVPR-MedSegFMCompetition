import os
import numpy as np
import cv2
np.random.seed(2025)
import cc3d
from skimage import segmentation
import copy
import multiprocessing as mp
import glob

def show_box_cv2(image, box, color=(255, 0, 0), thickness=2):
    """
    Draws a rectangle on an image using OpenCV.
    Args:
        image: The input image (numpy array).
        box: A bounding box, either 2D ([x_min, y_min, x_max, y_max]) or 3D ([x_min, y_min, z_min, x_max, y_max, z_max]).
        color: Color of the rectangle in BGR (default is blue).
        thickness: Thickness of the rectangle border (default is 2).
    Returns:
        The image with the rectangle drawn.
    """
    color = tuple(map(int, color))
    if len(box) == 4:  # 2D bounding box
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    else:  # 3D bounding box
        x_min, y_min, z_min, x_max, y_max, z_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

def show_mask_cv2(mask, image, color=None, alpha=0.5):
    assert mask.sum()>0
    if color is None:
        color = np.random.randint(0, 255, 3)
    h, w = mask.shape[-2:]
    overlay = np.zeros_like(image)
    for i in range(3):
        overlay[:, :, i] = color[i]
    overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
    combined = cv2.addWeighted(overlay, alpha, image, 1-alpha , 0)

    return combined

def mask2D_to_bbox(gt2D, file):
    try:
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        bbox_shift = np.random.randint(0, 6, 1)[0]
        scale_y, scale_x = gt2D.shape
        bbox_shift_x = int(bbox_shift * scale_x/256)
        bbox_shift_y = int(bbox_shift * scale_y/256)
        #print(f'{bbox_shift_x=} {bbox_shift_y=} with orig {bbox_shift=}')
        x_min = max(0, x_min - bbox_shift_x)
        x_max = min(W-1, x_max + bbox_shift_x)
        y_min = max(0, y_min - bbox_shift_y)
        y_max = min(H-1, y_max + bbox_shift_y)
        boxes = np.array([x_min, y_min, x_max, y_max])
        return boxes
    except Exception as e:
        raise Exception(f'error {e} with file {file}')


def mask3D_to_bbox(gt3D, file):
    b_dict = {}
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    # middle of z_indices
    z_middle = z_indices[len(z_indices)//2]
    D, H, W = gt3D.shape
    b_dict['z_min'] = z_min
    b_dict['z_max'] = z_max
    b_dict['z_mid'] = z_middle

    gt_mid = gt3D[z_middle]

    box_2d = mask2D_to_bbox(gt_mid, file)
    x_min, y_min, x_max, y_max = box_2d
    b_dict['z_mid_x_min'] = x_min
    b_dict['z_mid_y_min'] = y_min
    b_dict['z_mid_x_max'] = x_max
    b_dict['z_mid_y_max'] = y_max

    assert z_min == max(0, z_min)
    assert z_max == min(D-1, z_max)
    return b_dict

path = 'path-to-npz-files'
path_dest = 'destination-path'
os.makedirs(path_dest, exist_ok=True)
sanity_dir = os.path.join(path_dest, 'sanity')
os.makedirs(sanity_dir, exist_ok=True)
files = glob.glob(os.path.join(path, '*/*/*.npz'))
files = [x for x in files if 'Microscopy' not in x]
files = sorted(files)

print(f'number of files {len(files)}')

def process(file):
    print(f'processing file {file}')

    npz = np.load(file, allow_pickle=True)
    imgs = npz['imgs']
    
    gts = npz['gts']
    gts, _, _ = segmentation.relabel_sequential(gts)
    spacing = npz['spacing']
    unique_labs = np.unique(gts)[1:]

    boxes_list = []
    for lab in unique_labs:
        gt = gts==lab
        box_dict = mask3D_to_bbox(gt, file)
        boxes_list.append(box_dict)

    for j, box_dict in enumerate(boxes_list):
        color = np.random.randint(0, 255, 3)
        img_mid = imgs[box_dict['z_mid']].copy()
        img_mid = np.expand_dims(img_mid, axis=-1).repeat(3, axis=-1)
        box2D = [box_dict['z_mid_x_min'], box_dict['z_mid_y_min'], box_dict['z_mid_x_max'], box_dict['z_mid_y_max']]
        img_mid = show_box_cv2(img_mid, box2D, color=color, thickness=2)
        img_mid = show_mask_cv2((gts[box_dict['z_mid']]==unique_labs[j]).astype(np.uint8), img_mid.astype(np.uint8), color=color, alpha=0.5)
        cv2.imwrite(os.path.join(sanity_dir, os.path.basename(file).replace('.npz', f'_boxIdx{j}.png')), img_mid)

    assert gt.sum() > 0
    np.savez_compressed(os.path.join(path_dest, os.path.basename(file)), imgs=imgs, gts=gts, boxes=boxes_list, spacing=spacing)

if __name__ == '__main__':
    with mp.Pool(16) as p:
        p.map(process, files)
