from skimage import io
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage
from skimage.segmentation import find_boundaries
import pandas as pd


def fill_holes_in_cells(lb, is_plot = False):
    """
    This function fills holes inside cells within the labels segmentation mask

    Parameters
    ----------
    lb : 2D numpy array (int16)
        Cell labels mask with potential holes in cells.
    is_plot : bool, optional
        Plot the cells with holes for debugging. The default is False.

    Returns
    -------
    filled_segmentation : 2D numpy array (int16)
        Cell labels mask with filled holes inside cells.

    """
    # Get unique cell IDs (ignoring background, assumed to be 0)
    cell_ids = np.unique(lb)
    cell_ids = cell_ids[cell_ids != 0]  # Remove background ID if needed

    cells_with_holes = []

    cells_inside_cells = []

    # Copy of the segmentation image to modify
    filled_segmentation = lb.copy()

    # Loop through each unique cell ID
    for cell_id in cell_ids:
        # Create binary mask for the current cell
        cell_mask = (lb == cell_id).astype(np.uint8) * 255

        # Find external and internal contours
        contours, hierarchy = cv2.findContours(cell_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # If there are holes (nested contours)
        if hierarchy is not None:
            for i, contour in enumerate(contours):
                # If the contour is an inner hole (has a parent)
                if hierarchy[0][i][3] != -1:
                    # Create a mask for flood fill
                    hole_mask = np.zeros_like(lb, dtype=np.uint8)
                    
                    # Draw the hole contour on the mask
                    cv2.drawContours(hole_mask, [contour], -1, 255, thickness=cv2.FILLED)
                    
                    # Check if "hole" are a different cell
                    cell_id_of_hole = np.unique(lb[hole_mask.astype('bool')])
                    
                    # Remove background zero or cell_id
                    cell_id_of_hole = cell_id_of_hole[cell_id_of_hole != 0]
                    cell_id_of_hole = cell_id_of_hole[cell_id_of_hole != cell_id]
                    
                    
                    if cell_id_of_hole.size > 0:
                        cells_inside_cells.append(cell_id_of_hole)
                        

                    # Assign the cell ID to the hole pixels
                    filled_segmentation[hole_mask == 255] = cell_id
                    
                    # Print cell mask before and after
                    
                    if is_plot:
                        hole_mask_filled = np.zeros_like(filled_segmentation, dtype=np.uint8)
                        hole_mask_filled[filled_segmentation == cell_id] = 255
                        
                        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True , sharey=True)
                        axes[0].imshow(cell_mask)
                        axes[1].imshow(hole_mask_filled)
                    
                    # Add cell_id to list
                    
                    cells_with_holes.append(cell_id)

    return filled_segmentation
                    
 
def merge_segmentation_masks(lb1 , lb2 , is_plot = False):
    """
    This function merges two segmentation masks into one. 
    Specifically it takes a mask of large cells (goblet cells) and merges it with a segmentation of mornal cells.

    Parameters
    ----------
    lb1 : 2D numpy array (int16)
        Cell labels mask for the normal cell size (smaller)
    lb2 : 2D numpy array (int16)
        Cell labels mask for the larger cell size (goblet cells)
    is_plot : bool, optional
        Plot images for debugging.

    Returns
    -------
    lb_merged_ordered : 2D numpy array (int16)
        Merged labels mask.
    bd_merged : 2D numpy array (uint8)
        Merged cell boundaries.

    """
        
    
    #----- fill in cells with holes or cells insode cells -----#
    lb1 = fill_holes_in_cells(lb1)
    lb2 = fill_holes_in_cells(lb2)
    
    
    #new labels file
    lb1_new = np.copy(lb1)
    lb2_new = np.copy(lb2)
    
    #number of cells in each segmentation mask
    n_lb1 = np.max(lb1)
    
    #lb2 cell intersected with lb1
    intersection_mask = np.logical_and(lb1, lb2)
    lb1_inter = np.unique(intersection_mask * lb1)
    lb2_inter = np.unique(intersection_mask * lb2)
    
    #remove '0' from cell list (background)
    lb1_inter = np.delete(lb1_inter , 0)
    lb2_inter = np.delete(lb2_inter , 0)
    
   #----- remove cells that are overlap in more than 50% (keep larger cells) -----#
    for c1 in lb1_inter:
        #mask c1
        c1_mask = np.copy(lb1)
        c1_mask[lb1 != c1] = 0
        c1_mask[lb1 == c1] = 1
        
        c1_erroded = ndimage.binary_erosion(c1_mask , iterations = 2)
        
        #intersecting lb2 cells in c1
        lb2_in_c1 = np.unique(c1_mask * lb2)
        lb2_in_c1 = np.delete(lb2_in_c1 , 0)
        for c2 in lb2_in_c1:
            #mask c2
            c2_mask = np.copy(lb2)
            c2_mask[lb2 != c2] = 0
            c2_mask[lb2 == c2] = 1
            
            #criterias
            if np.sum(np.logical_or(c1_mask, c2_mask)) == 0:
                jaccard_ind = 0
                
            else:
                jaccard_ind = np.sum(np.logical_and(c1_mask, c2_mask)) / np.sum(np.logical_or(c1_mask, c2_mask))
                
            if np.min([np.sum(c1_mask) , np.sum(c2_mask)]) == 0:
                meet_min = 0
                meet_min_erroded = 0
            else: 
                meet_min = np.sum(np.logical_and(c1_mask, c2_mask)) / np.min([np.sum(c1_mask) , np.sum(c2_mask)])
                meet_min_erroded = np.sum(np.logical_and(c1_erroded, c2_mask)) / np.min([np.sum(c1_erroded) , np.sum(c2_mask)])

            if (jaccard_ind > 0.5) or ((jaccard_ind > 0.25) and (meet_min > 0.95)):
                    lb2_new[lb2_new == c2] = 0
            elif (meet_min_erroded > 0.95): 
                if np.sum(c1_mask) > np.sum(c2_mask):
                    lb2_new[lb2_new == c2] = 0
                else:
                    lb1_new[lb1_new == c1] = 0
                    
                #plot overlaying cells 
                if is_plot:
                    fig , ax = plt.subplots()
                    ax.imshow(c1_mask , cmap = 'Blues')
                    ax.imshow(c2_mask , cmap = 'Reds' , alpha=0.5)
                    text = "Cell1  : %d \n Jaccard index = %.2f\n Meet-min = %.2f \n Meet-min-erroded = %.2f" %(c1 , np.round(jaccard_ind ,2) , np.round(meet_min ,2) , np.round(meet_min_erroded,2))
                    ax.annotate(text,
                    xy=(0.5, 0.7), xytext=(0, 10),
                    xycoords=('axes fraction', 'figure fraction'),
                    textcoords='offset points',
                    size=14, ha='center', va='bottom')
                
    #----- keep pixels belong to normal cells and remove from goblet cells  -----#          
    intersection_mask = np.logical_and(lb1_new, lb2_new)
    lb1_new[intersection_mask] = 0
    
    
    #----- smooth edges for goblet cells using oppening operator -----#
    
    diameter = 10
    radius = diameter // 2
    x = np.arange(-radius, radius+1)
    x, y = np.meshgrid(x, x)
    r = x**2 + y**2
    se = r < radius**2
    lb1_opened_mask = ndimage.binary_opening(lb1_new, se)
    lb1_opend = lb1_opened_mask * lb1_new
    
    #mask normal cells for display
    lb2_mask = np.copy(lb2_new)
    lb2_mask[lb2_new > 0] = 1
    
    if is_plot:
        fig , axs = plt.subplots(1,3 , sharex=True , sharey=True)
        axs[0].imshow(lb1 , cmap = 'rainbow')
        axs[0].imshow(lb2_mask , cmap = 'gray' , alpha = 0.5)
        axs[0].set_title('after removing overlapping cells')
        axs[1].imshow(lb1_new , cmap = 'rainbow')
        axs[1].imshow(lb2_mask , cmap = 'gray' , alpha = 0.5)
        axs[1].set_title('after masking with normal cells')
        axs[2].imshow(lb1_opend , cmap = 'rainbow')  
        axs[2].imshow(lb2_mask , cmap = 'gray' , alpha = 0.5)
        axs[2].set_title('after smoothing edges using opening operator')
         
        plt.figure()
        plt.imshow(lb1_new , cmap = 'Blues')
        plt.imshow(lb1_opend , cmap = 'Reds' , alpha = 0.5)
    
    #----- if cell is splited into multuple parts -> take the part with the maximal area -----#
    
    cells_in_lb1 = np.unique(lb1_opend)
    cells_in_lb1 = np.delete(cells_in_lb1 , 0)
    
    lb1_final = np.zeros_like(lb1_opend)
    for c in cells_in_lb1:
        #mask one cell in lb1
        c_mask = np.copy(lb1_opend)
        c_mask[lb1_opend != c] = 0
        c_mask[lb1_opend == c] = 1
        c_mask = c_mask.astype('uint8')
        
        # apply connected component analysis to cell
        output = cv2.connectedComponentsWithStats(c_mask, connectivity = 4)
        (numLabels, labels, stats, _) = output
        
        #first component is always background
        if numLabels > 1:
            # remove background component
            stats = stats[1:]
            
            label_to_keep = np.argmax(stats[:, -1])
            c_mask_new = np.zeros_like(c_mask)
            
            if np.max(stats[:, -1]) >= 10:
               c_mask_new[labels == label_to_keep + 1] = 1
            
            if numLabels > 2 and is_plot:
                #plot before and after detecting the largest component
                if is_plot:
                    plt.figure()
                    plt.imshow(c_mask , cmap = 'Blues')
                    plt.imshow(c_mask_new , cmap = 'Reds' , alpha = 0.5)
            
            lb1_final[c_mask_new == 1] = c
            
    #----- merege lb1 and lb2 -----#
    
    #number of cells in each segmentation mask
    n_lb1 = np.max(lb1_final)
    
    #check there are no ovelaping pixels
    sanity_check = np.sum(np.logical_and(lb1_final, lb2_new))
    
    #change lb2 numbering to start from n_lb1+1
    lb2_new = lb2_new + n_lb1
    lb2_new[lb2_new == n_lb1] = 0
    
    lb_merged = lb1_final + lb2_new
    lb_merged = lb_merged.astype('int16')
    
    #order cells in consecutive order
    
    order_cell_dict = dict(zip(np.unique(lb_merged), np.arange(0 , len(np.unique(lb_merged)))))
    
    lb_merged_ordered = np.reshape(pd.Series(lb_merged.flatten()).map(order_cell_dict).to_numpy(dtype='int16') , lb_merged.shape)
    
    sanity_check_2 = len(np.unique(lb_merged_ordered)) == np.max(lb_merged_ordered) + 1
    
    #border mask
    bd_merged = find_boundaries(lb_merged_ordered, mode='inner').astype(np.uint8)
    bd_merged[bd_merged > 0] = 255    
    
    return lb_merged_ordered , bd_merged


if __name__ == "__main__":
   
    labels1_path = 'fill in your path' 
    labels2_path = 'fill in your path' 

    #-----read segmentaions files -----#

    lb1 = io.imread(labels1_path)
    lb2 = io.imread(labels2_path)
    
    segmentation_labels_merged  , segmentation_borders_merged = merge_segmentation_masks(lb1 , lb2 , is_plot = False)
    
    #-----plot the labels mask before and after the merge-----#
    
    img_to_plot = [lb1 , lb2 ,segmentation_labels_merged ]
    img_titles = ['label mask 1' , 'label mask 2' , 'merged label mask']
    
    fig, axes = plt.subplots(1,3 , sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, lb_img in enumerate(img_to_plot):
        
        num_labels = len(np.unique(lb_img))
        
        # Initialize color array with random RGB colors
        colors = np.random.randint(0, 256, size=(num_labels, 3), dtype=np.uint8)
        
        # Set label 0 to black
        colors[0] = [0, 0, 0]
        
        # Map each label in the mask to its corresponding color
        colored_image = colors[lb_img]
        
        axes[i].imshow(colored_image)
        axes[i].set_title(img_titles[i])
        
        axes[i].axis('off')

    
