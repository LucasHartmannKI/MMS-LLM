import numpy as np
import os
import json
from sklearn.cluster import DBSCAN
import open3d as o3d
import pandas as pd
import csv
import pickle 
#import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
#from tqdm import tqdm
import gc  # For garbage collection
import traceback


def DBS(xyz, eps, min_samples):
    # Perform DBSCAN clustering for instance segmentation on the current label's point cloud
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Adjust these parameters as needed
    instance_labels = dbscan.fit_predict(xyz)
    combined_data = np.column_stack((xyz, instance_labels))
    
    return combined_data
    

def instance_segment(main_folder, para):
    
    target_points = 8192 
    point_clouds_list = []
    labels_list = []
    ########################################################################################################
    ########################################(ajust label id)################################################
    #c5:['7','8','13','15','20'] c8:['4','5','7','8','13','15','20','25']
    # label 25 is tree, i have mergerd     9.0: "treetop"  and   10.0: "tree trunk",
    for folder_name in ['4','5','7','8','13','15','20','25']: #select the label id ,what you want to handel.
        #print(folder_name)
        sub_folder = os.path.join(main_folder, folder_name)
        key = float(folder_name)
        eps = para[key]['eps']
        min_samples = para[key]['min_samples']
        Threshold = para[key]['Threshold']
        name = para[key]['name']
        list_file =os.listdir(sub_folder)
        i = 1
        for file in list_file:
            if  file.endswith('.csv'):  
                try:
                    file_path = os.path.join(sub_folder, file)
                    data= pd.read_csv(file_path)
                    data =  data.to_numpy()
                    xyz = data[:, 0:3]

                    if xyz.shape[0] == 0:
                        print(f"Skipping empty file: {file}")
                        continue
                    
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Adjust these parameters as needed
                    labels = dbscan.fit_predict(xyz)
                    instance_labels = np.unique(labels)
                    
                    for label in instance_labels:
                        
                        if label == -1 :  # Skip the noise label (label -1)
                            continue   
                        label_mask = (labels == label)
                        instance = data[label_mask]
                        #posi = np.mean(xyz[label_mask], axis=0)
                        num_points = len(instance)
                        label_asnumpy = np.array([label], dtype=np.float32)
                        if num_points>=Threshold:
  
                            if num_points < target_points:
                                # Upsampling(repeat)
                                indices = np.random.choice(num_points, target_points - num_points, replace=True)
                                upsampled_points = instance[indices]
                                instance = np.vstack((instance, upsampled_points))

                            # elif num_points > target_points:         # This is not necessary.
                            #     # Downsampling
                            #     pcd = o3d.geometry.PointCloud()
                            #     pcd.points = o3d.utility.Vector3dVector(instance[:, :3])
                            #     downsampled_pcd = pcd.farthest_point_down_sample(target_points)
                            #     instance = np.asarray(downsampled_pcd.points)
                            # normal
                            
                            #xyz(-1-1)
                            xyz = instance[:,:3]  
                            xyz_center = np.mean(xyz, axis=0)
                            xyz_norm = xyz - xyz_center
                            max_distance = np.max(np.sqrt(np.sum(xyz_norm ** 2, axis=1)))
                            xyz_norm /= max_distance
                            
                            #reflectivity(-1-1)
                            reflectivity = instance[:,3]
                            reflectivity_min = reflectivity.min(axis=0)
                            reflectivity_max = reflectivity.max(axis=0)
                            reflectivity = (reflectivity - reflectivity_min) / (reflectivity_max - reflectivity_min)
                            reflectivity = np.repeat(reflectivity[:,np.newaxis], 3, axis =1)   
                            
                            #rgb(0-1)
                            rgb = instance[:,4:] 
                            rgb/=255.0
                            
                            instance1 = np.hstack((xyz,rgb)).astype(np.float32)
                            instance2 = np.hstack((xyz_norm,reflectivity)).astype(np.float32)
                            instance3 = np.hstack((xyz_norm,rgb)).astype(np.float32)
                            
 
                            folder1 = f"../data/instance_pointcloud/instance_seg_c8_xyz_orignal_txt/{name}"
                            folder2 = f"../data/instance_pointcloud/instance_seg_c8_reflectivity_txt/{name}"
                            folder3 = f"../data/instance_pointcloud/instance_seg_c8_rgb_txt/{name}"
                            folder4 = f"../data/instance_pointcloud/instance_seg_c8_reflectivity_npy/{name}"
                            for folder in [folder1, folder2, folder3, folder4]:
                                if not os.path.exists(folder):
                                    os.makedirs(folder)

                            outputfile1 = os.path.join(folder1, f"{name}_{i:04}.txt")
                            outputfile2 = os.path.join(folder2, f"{name}_{i:04}.txt")
                            outputfile3 = os.path.join(folder3, f"{name}_{i:04}.txt")
                            outputfile4 = os.path.join(folder4, f"{name}_{i:04}.npy")
                            np.savetxt(outputfile1, instance1, fmt="%.5f") 
                            np.savetxt(outputfile2, instance2, fmt="%.5f") 
                            np.savetxt(outputfile3, instance3, fmt="%.5f") 
                            np.save(outputfile4, instance2)  

                            i+=1
 
                        
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    traceback.print_exc()
                     
    return
    
#eps:the maximum distance between two points/ min_samples:controls the minimum number of points needed to form a cluster
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
    #instacne segment
    main_folder = '../data/semantic_seg'    #####################
    instance_segment(main_folder, para)