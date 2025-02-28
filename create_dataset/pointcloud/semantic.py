import numpy as np
import os
import json
from plyfile import PlyData, PlyElement
import gc  # For garbage collection
import traceback

def load_ply(file):
    plydata = PlyData.read(file)
    
    # Directly process ply data, avoiding unnecessary list operations
    feat_values = [np.asarray(plydata.elements[0].data[feat.name]) for feat in plydata.elements[0].properties]
    
    # Use np.column_stack instead of np.stack, and slice the necessary data
    points = np.column_stack((feat_values[0], feat_values[1], feat_values[2]))  # x, y, z coordinates
    features = np.column_stack((feat_values[4], feat_values[5], feat_values[6], feat_values[7]))  # Reflectivity and RGB
    labels = feat_values[8]  # Labels
    
    return points, labels, features

    
def semantic_segment(Lidar_folder, para):
    # Get all the PLY files in the Lidar_folder
    files = [f for f in os.listdir(Lidar_folder) if f.endswith('.ply')]
    for idx, file in enumerate(files):
        try:
            ply_file = os.path.join(Lidar_folder, file)
            points, labels, features = load_ply(ply_file)
            # Unique labels in the dataset
            semantic_labels = np.unique(labels)
    
            # Iterate over each unique label and segment the data
            for label in semantic_labels:
                # Extract points belonging to the current label
                label_mask1 = labels == label
                xyz = points[label_mask1]
                feature = features[label_mask1]  
                segment = np.column_stack((xyz, feature))
                # Create a Lidar_folder to save the output files
                output_folder = f"../data/semantic_seg/{label}"
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                # Create the output file path
                output_file = os.path.join(output_folder, f"{label}_{idx}.csv")
                
                # Save the combined data to a CSV file for each label
                np.savetxt(output_file, segment, delimiter=',', header="x,y,z,reflectivity,r,g,b", comments="", fmt='%f')
            # Clear variables to free up memory after processing each file
            del points, labels, features, xyz, feature, segment
            gc.collect()  # Trigger garbage collection to free memory
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            traceback.print_exc()   
    return
# The parameters for different classes
para = {
    1.0: {'eps': 0.5, 'min_samples': 10,'Threshold':300,'name':'road'},  # road &&
    2.0: {'eps': 0.5, 'min_samples': 10,'Threshold':300,'name':'sidewalk'},  # sidewalk &&
    3.0: {'eps': 0.5, 'min_samples': 10,'Threshold':300,'name':'building'},  # building &&1007
    4.0: {'eps': 0.5, 'min_samples': 8,'Threshold':10000,'name':'wall'},  # wall %
    5.0: {'eps': 0.5, 'min_samples': 10,'Threshold':3000,'name':'fence'},  # fence %
    6.0: {'eps': 0.5, 'min_samples': 10,'Threshold':1000,'name':'pole'},  # pole %
    7.0: {'eps': 0.5, 'min_samples': 10,'Threshold':1000,'name':'traffic_light'},  # traffic light/compl
    8.0: {'eps': 0.5, 'min_samples': 10,'Threshold':1000,'name':'traffic_sign'},  # traffic sign
    9.0: {'eps': 0.5, 'min_samples': 10,'Threshold':3000,'name':'treetop'},  # treetop %
    10.0: {'eps': 0.5, 'min_samples': 10,'Threshold':1000,'name':'tree_trunk'},  # tree trunk %
    11.0: {'eps': 0.5, 'min_samples': 10,'Threshold':3000,'name':'bush'},  # bush/hedge %
    12.0: {'eps': 0.5, 'min_samples': 10,'Threshold':3000,'name':'terrain'},  # terrain %
    13.0: {'eps': 0.3, 'min_samples': 10,'Threshold':300,'name':'person'},  # person 
    14.0: {'eps': 0.5, 'min_samples': 10,'Threshold':300,'name':'rider'},  # rider -
    15.0: {'eps': 0.35, 'min_samples': 8,'Threshold':8000,'name':'car'},  # car
    16.0: {'eps': 0.5, 'min_samples': 10,'Threshold':15000,'name':'truck'},  # truck -
    17.0: {'eps': 0.5, 'min_samples': 10,'Threshold':10000,'name':'bus'},  # bus -
    18.0: {'eps': 0.5, 'min_samples': 10,'Threshold':10000,'name':'train'},  # train -
    19.0: {'eps': 0.3, 'min_samples': 10,'Threshold':500,'name':'motorcycle'},  # motorcycle -
    20.0: {'eps': 0.2, 'min_samples': 10,'Threshold':500,'name':'bicycle'},  # bicycle
    21.0: {'eps': 0.5, 'min_samples': 10,'Threshold':500,'name':'movable_objects'},  # movable objects -
    22.0: {'eps': 0.5, 'min_samples': 10,'Threshold':500,'name':'construction site'},  # construction site -
    23.0: {'eps': 0.5, 'min_samples': 10,'Threshold':500,'name':'undefined_static'},  # undefined static -
    24.0: {'eps': 0.5, 'min_samples': 10,'Threshold':500,'name':'undefined_mixed'},  # undefined mixed -
    25.0: {'eps': 0.5, 'min_samples': 10,'Threshold':3000,'name':'tree'},  # tree
}

if __name__ == "__main__":
    Lidar_folder = "../data//kpconv-seg-vanila" # Here you need to change the path to the folder where your Lidar-pointcloud are stored.
    semantic_segment(Lidar_folder, para)  