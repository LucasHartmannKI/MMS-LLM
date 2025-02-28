import numpy as np
import matplotlib.pyplot as plt
import os
import zarr
import pyproj
import json
import rawpy
from scipy.ndimage import binary_dilation, gaussian_filter, binary_fill_holes
import cv2
from scipy.ndimage import binary_fill_holes

#parameters-----------------------------------
base_path = "/data/large"
project_path = "20221006_LINDENNORD_7"
tileordered_path = "tileordered_scandata/tileordered.zarr"
image_info_path = "imagedata/original_image_orientations.json"
tileordered = zarr.open(os.path.join(base_path, project_path, tileordered_path))
level = 21#20
tile_table = tileordered[f"tile_tables/{level:02d}"][:].astype(int)
object_folder = "creat_dataset/data/instance_pointcloud/instance_seg_c17_xyz_original_txt"
save_folder = "creat_dataset/data/image_original_filter2000"
use_mask = False
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
#functions-----------------------------------
def latlon2web(lat, lon, level):
    f = 1 << level
    x = (lon + 180.0) * f / 360.0
    y = (np.pi - np.log(np.tan(np.pi/4 + np.deg2rad(lat)/2))) * f / (2 * np.pi)
    return x, y

def get_points_on_image_plane(points, camera_position, camera_matrix, io_params):
    # Points in camera system.
    cmcs_points =  (points - camera_position) @ camera_matrix.T
    # Take only points in front of the camera.
    cmcs_points = cmcs_points[cmcs_points[:,2] > 0]

    # Project onto image plane.
    img_points = np.empty((cmcs_points.shape[0], 2))
    img_points[:,0] = (cmcs_points[:,0] / cmcs_points[:,2]) * io_params["fx"] + io_params["cx"]
    img_points[:,1] = (cmcs_points[:,1] / cmcs_points[:,2]) * io_params["fy"] + io_params["cy"]

    # Only keep points inside image.
    mask = np.logical_and(np.logical_and(np.logical_and(
        img_points[:,0] >= 0,  img_points[:,0] <= io_params["nx"]),
        img_points[:,1] >= 0), img_points[:,1] <= io_params["ny"])
    return img_points[mask]

def compute_undistorted_image_coordinates(points, io_params):
# Use the definitions in the Riegl doc.
    x = (points[:,0] - io_params["cx"]) / io_params["fx"]
    y = (points[:,1] - io_params["cy"]) / io_params["fy"]
    xx = x*x
    xy = x*y
    yy = y*y
    # An alternative (buggy) formula from Riegl would be:
    # np.arctan(np.sqrt(xx + yy))
    r2 = xx + yy
    r4 = r2 * r2
    r6 = r2 * r4
    r8 = r2 * r6
    du = io_params["fx"] * (
         x * (io_params["k1"] * r2 +\
              io_params["k2"] * r4 +\
              io_params["k3"] * r6 +\
              io_params["k4"] * r8) +\
         2 * io_params["p1"] * x * y +\
         io_params["p2"] * (r2 + 2 * xx))
    dv = io_params["fy"] * (
         x * (io_params["k1"] * r2 +\
              io_params["k2"] * r4 +\
              io_params["k3"] * r6 +\
              io_params["k4"] * r8) +\
         2 * io_params["p2"] * x * y +\
         io_params["p1"] * (r2 + 2 * yy))
    return points + np.vstack((du, dv)).T


def extract_object_with_dense_mask(img, img_points, dilation_iters, blur_sigma):

    # initial mask
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for x, y in img_points.astype(int):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            mask[y, x] = 1

    # Expansion mask to fill gaps
    mask = binary_dilation(mask, iterations=dilation_iters).astype(np.uint8)
    
    # Gaussian blur to smooth edges
    mask = gaussian_filter(mask.astype(float), sigma=blur_sigma)
    mask = (mask > 0.5).astype(np.uint8)  
    mask = binary_fill_holes(mask).astype(np.uint8)
    
    # apply mask
    masked_image = img * mask[:, :, None]

    coords = np.where(mask > 0)
    y3_min, y3_max = np.min(coords[0]), np.max(coords[0])
    x3_min, x3_max = np.min(coords[1]), np.max(coords[1])

    cropped_image = best_img[y3_min:y3_max+1, x3_min:x3_max+1]
    return cropped_image

def extract_object_with_convex_hull(img, img_points):

        
    img_points = np.array(img_points, dtype=np.float32)

    hull = cv2.convexHull(img_points)
    #hull = hull.reshape(-1, 2)
    hull = hull.reshape(-1, 2).astype(np.int32)
    circled_image = img.copy()

    cv2.polylines(circled_image, [hull], isClosed=True, color=(0, 255, 0), thickness=2)  # green

    # Create a mask for the convex hull
    # mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # cv2.fillConvexPoly(mask, hull, 255)

    # coords = np.column_stack(np.where(mask > 0))
    # y_min, x_min = coords.min(axis=0)
    # y_max, x_max = coords.max(axis=0)

    # cropped_image = circled_image[y_min:y_max+1, x_min:x_max+1]
    # cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
    return circled_image


#main function-----------------------------------
def main():
    #best_mask = None  
    img = None
    for sub_folder in os.listdir(object_folder):
        sub_folder_path = os.path.join(object_folder, sub_folder)
        for file in os.listdir(sub_folder_path):
            #load instance pointcloud
            
            if file.endswith(".txt"):
                name = file.split(".")[0]
                object_file = os.path.join(sub_folder_path, file)
            pointcloud = np.loadtxt(object_file, delimiter=' ')
            pointcloud = pointcloud[:,:3]
    
            #get bounding box
            x1_min, x1_max = np.min(pointcloud[:,0]),np.max(pointcloud[:,0])
            y1_min, y1_max = np.min(pointcloud[:,1]),np.max(pointcloud[:,1])
            bbox = np.array([[x1_min, y1_min],  # Lower left.
                     [x1_max, y1_max]]) # Upper right.
            
            #transform bounding box to web mercator
            transformer = pyproj.Transformer.from_crs("epsg:25832", "epsg:4326")
            bbox_wgs84 = np.empty(bbox.shape, dtype=float)
            bbox_wgs84[:,0], bbox_wgs84[:,1] = transformer.transform(bbox[:,0], bbox[:,1])
            bbox_web = np.array([latlon2web(*row, level) for row in bbox_wgs84])
            bbox_tiles = np.floor(bbox_web).astype(int)
            # These are the bounds for the tile indices.
            tllx, turx = min(bbox_tiles[:,0]), max(bbox_tiles[:,0])
            tlly, tury = min(bbox_tiles[:,1]), max(bbox_tiles[:,1])
    
            #get zone point cloud
            mask = np.logical_and(np.logical_and(np.logical_and(
            tile_table[:,0] >= tllx,  tile_table[:,0] <= turx),
            tile_table[:,1] >= tlly), tile_table[:,1] <= tury)
            points = []
            for tix, tiy, i_start, i_end in tile_table[mask]:
                #print(f"Grabbing tile ({tix},{tiy}): {i_end-i_start} points")
                points.append(tileordered["xyz_proj"][i_start:i_end])
            points = np.concatenate(points)
    
            #load camera locations
            x2_min, x2_max = np.min(points[:,0]),np.max(points[:,0])
            y2_min, y2_max = np.min(points[:,1]),np.max(points[:,1])
            bbox_points = np.array([[x2_min, y2_min],  # Lower left.
                            [x2_max, y2_max]]) # Upper right.
            # Preliminary... later this will be a path in the repository.
            image_orientations_path = os.path.join(base_path, project_path, image_info_path)
            with open(image_orientations_path) as f:
                image_orientations = json.load(f)
            # Get all camera locations inside my UTM bounding box.
            # For this example, I am interested in "Camera 4" only.
            locations = np.array(image_orientations["cameras"]["Camera 1"]["images"]["proj_pos"])
            mask = np.logical_and(np.logical_and(np.logical_and(
                locations[:,0] >= bbox_points[0,0],  locations[:,0] < bbox[1,0]),
                locations[:,1] >= bbox_points[0,1]), locations[:,1] < bbox[1,1])
            # This is the interesting indices.
            # In this example, it is just the first n images.
            indices = np.nonzero(mask)[0]
            if len(indices) == 0:
                print("No images in this bounding box.")
                continue
    
            #select a image
            best_img = None 
            num_points = 0
            for i in range(len(indices)):
                index_in_sublist = i#5  # May use 0, 1, 2, ... up to length of indices -1.
                image_index = indices[index_in_sublist]
                image_path = os.path.join(
                    image_orientations["cameras"]["Camera 1"]["path"],
                    image_orientations["cameras"]["Camera 1"]["images"]["files"][image_index])
                #print(image_path)
                # Check this image.
                if not os.path.exists(image_path):
                    print(f"File not found: {image_path}")
                    continue
                try:
                    with rawpy.imread(image_path) as raw:
                            img = raw.postprocess()
                except rawpy.LibRawFileUnsupportedError as e:
                    print(f"Unsupported RAW file format: {image_path}")
                    continue
                except rawpy.LibRawIOError as e:
                    print(f"Error reading file {image_path}: {e}")
                    continue
                    
                proj_pos = np.array(
                image_orientations["cameras"]["Camera 1"]["images"]["proj_pos"][image_index])
                proj_mat = np.array(
                    image_orientations["cameras"]["Camera 1"]["images"]["proj_mat"][image_index]).reshape(3,3)
                # Get the interior orientation parameters.
                io_params_names = ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2", "nx", "ny"]
                io_params = { k: float(image_orientations["cameras"]["Camera 1"]["parameters"][k])
                            for k in io_params_names}
                img_points = get_points_on_image_plane(
                    pointcloud, proj_pos, np.linalg.pinv(proj_mat), io_params)
                new_num_points = img_points.shape[0]

                if new_num_points <= 2000:
                    print(f"the points in image are less than 2000 {image_index}")
                    continue
                elif new_num_points > num_points:
                    num_points = new_num_points
                    undist_img_points = compute_undistorted_image_coordinates(img_points, io_params) 
                    if use_mask:
                        image_masked = extract_object_with_dense_mask(img, undist_img_points, dilation_iters=10, blur_sigma=5)
                        best_img = image_masked
  
                    else:
                        try:
                            circled_image= extract_object_with_convex_hull(img, img_points)
                        except ValueError as e: 
                            print(f"Error processing image {image_index}: {e}")
                            continue
                        #circled_image = extract_object_with_convex_hull(img, img_points)
                        best_img = circled_image
                        
    
            #plt.figure(figsize=(8, 8))
            #plt.title("Cropped Object")
            if best_img is not None:
                plt.axis("off")
                save_path = os.path.join(save_folder, f"{name}.png")
                print('save {name}.png')
                plt.imsave(save_path, best_img)
                plt.close()


    return
    
if __name__ == "__main__":
    main()
