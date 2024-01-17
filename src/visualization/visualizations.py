import torch 
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union
from matplotlib.patches import Patch

##-- Global Parameters --##

color_mapping = {
    0: (255, 0, 0),       # Red
    1: (0, 255, 0),       # Green
    2: (0, 0, 255),       # Blue
    3: (255, 255, 0),     # Yellow
    4: (255, 0, 255),     # Magenta
    5: (0, 255, 255),     # Cyan
    6: (255, 165, 0),     # Orange
    7: (128, 0, 128),     # Purple
    8: (0, 128, 0),       # Dark Green
    9: (128, 0, 0),       # Maroon
    10: (0, 128, 128),    # Teal
    11: (128, 128, 0),    # Olive
    12: (184, 134, 11),   # Dark Goldenrod
    13: (138, 43, 226),   # Blue Violet
    14: (165, 42, 42),    # Brown
    15: (95, 158, 160),   # Cadet Blue
    16: (255, 105, 180),  # Hot Pink
    17: (50, 205, 50),    # Lime Green
    18: (153, 50, 204),   # Dark Orchid
    19: (0, 206, 209),    # Dark Turquoise
    20: (220, 20, 60),    # Crimson
    21: (70, 130, 180),   # Steel Blue
    22: (46, 139, 87),    # Sea Green
    23: (255, 69, 0),     # Red-Orange
    24: (0, 100, 0),      # Dark Green
    25: (128, 128, 0),    # Olive
    26: (128, 0, 128),    # Purple
    27: (0, 139, 139),    # Dark Cyan
    28: (210, 105, 30),   # Chocolate
    29: (255, 165, 79)    # Peach
}


def visualize_segmentation(
                image: Union[torch.Tensor, np.array],
                   segmentation: Optional[Union[torch.Tensor, np.array]],
                    ground_truth_segmentation: Optional[Union[torch.Tensor, np.array]]=None,
                     class_labels: Optional[list] = None):

    def _add_img(img, ax):
        ax.imshow(img)
        ax.axis('off')
    def _map_segmentation(img):

        segImg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint)
        for id in range(classes):
            mask = (img == id)
            segImg[mask.squeeze()] = color_mapping[id]
        return segImg

    inclGT = False
    cols = 2
    rows = image.shape[0]
    classes = np.max(segmentation) if class_labels is None else len(class_labels)

    if ground_truth_segmentation is not None:
        assert image.shape[0] == ground_truth_segmentation.shape[0], "Must provide the same number of images and ground truth segmentations"
        inclGT = True
        cols = 3

    fig = plt.figure()

    ind = 1
    for i in range(image.shape[0]):
        img = image[i]
        ax = fig.add_subplot(rows, cols, ind)
        _add_img(img, ax)
        ind += 1
        if i == 0:
            ax.set_title(f"Original Image")

        seg = segmentation[i]
        seg = _map_segmentation(seg)
        ax = fig.add_subplot(rows, cols, ind)
        _add_img(seg, ax)
        ind += 1
        if i == 0:
            ax.set_title(f"Segmentation")

        if inclGT:
            gt_seg = ground_truth_segmentation[i]
            gt_seg = _map_segmentation(gt_seg)
            ax = fig.add_subplot(rows, cols, ind)
            _add_img(gt_seg, ax)
            ind += 1
            if i == 0:
                ax.set_title(f"Ground Truth")

    if class_labels is not None:
        legend_patches = [Patch(color=(rgb_values[0]/255,rgb_values[1]/255,rgb_values[2]/255,1.0), label=f'Index {index}') for index, rgb_values in list(color_mapping.items())[:classes]]
        fig.legend(handles=legend_patches, bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0., ncol=5)
    plt.tight_layout()
    return fig