import os
import numpy as np
import matplotlib.pyplot as plt


def rotate_point_cloud(point_cloud, axis, angle):
    """
    Rotate the point cloud around a specific axis by a given angle.
    Args:
        point_cloud: numpy array of shape (N, 3) or (N, >=3), the point cloud data.
        axis: str, the axis to rotate around ('x', 'y', or 'z').
        angle: float, rotation angle in radians.
    Returns:
        Rotated point cloud.
    """
    rotation_matrix = None
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
    
    # Apply rotation to the 3D coordinates
    rotated_points = point_cloud[:, :3] @ rotation_matrix.T
    return np.hstack([rotated_points, point_cloud[:, 3:]]) if point_cloud.shape[1] > 3 else rotated_points


def plot_8_views(name, point_cloud, use_reflexity, save_dir, resolution=512, show_origin=True):
    """
    Generate and save projections for 8 views of the point cloud.
    Args:
        point_cloud: np.array, the point cloud data.
        use_reflexity: bool, whether to use reflexity as color.
        save_dir: str, directory to save the images.
        resolution: int, resolution of the output images (square).
        show_origin: bool, whether to mark the origin on plots.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define 8 camera angles
    view_angles = [
        ("view0", -np.pi / 2, 0, 0),         # 
        ("view1", -np.pi / 2, 0, np.pi / 4), #
        ("view2", -np.pi / 2, 0, np.pi / 2), # 
        ("view3", -np.pi / 2, 0, np.pi * 3/ 4), # 
        ("view4", -np.pi / 2, 0, np.pi), # 
        ("view5", -np.pi / 2, 0, np.pi*5 /4), # 
        ("view6", -np.pi / 2, 0, np.pi*3/2), # 
        ("view7", -np.pi / 2, 0, np.pi * 7 /4 )  # 
    ]

    i=0
    for view_name, rot_x, rot_y, rot_z in view_angles:
        rotated_cloud = point_cloud.copy()

        # Apply rotations
        if rot_z != 0:
            rotated_cloud = rotate_point_cloud(rotated_cloud, 'z', rot_z)
        if rot_x != 0:
            rotated_cloud = rotate_point_cloud(rotated_cloud, 'x', rot_x)
        if rot_y != 0:
            rotated_cloud = rotate_point_cloud(rotated_cloud, 'y', rot_y)

        # Project to 2D (XY plane as an example for each view)
        xy_points = rotated_cloud[:, :2]
        
        # Color handling
        if use_reflexity:
            reflexity = point_cloud[:, 3]
            norm_reflexity = (reflexity - np.min(reflexity)) / (np.max(reflexity) - np.min(reflexity))
            colors = np.stack([norm_reflexity, norm_reflexity, norm_reflexity], axis=-1)
        else:
            colors = point_cloud[:, 3:6] if point_cloud.shape[1] >= 6 else 'blue'

        # Plot and save
        fig, ax = plt.subplots(figsize=(resolution / 100, resolution / 100))
        ax.scatter(xy_points[:, 0], xy_points[:, 1], s=1, c=colors)
        # if show_origin:
        #     ax.plot(0, 0, 'ro', markersize=5, label="Origin")
        #     ax.legend()
        #ax.set_title(f"Projection: {view_name}")
        ax.axis('equal')
        #ax.set_xlabel("X-axis")
        #ax.set_ylabel("Y-axis")
        ax.set_xticks([])  # 
        ax.set_yticks([])  # 

        # Save image to corresponding directory
        view_dir = os.path.join(save_dir, f"Cap3D_imgs_{view_name}")
        if not os.path.exists(view_dir):
            os.makedirs(view_dir)
        file_path = os.path.join(view_dir, f"{name}-{i}.png")
        fig.savefig(file_path, dpi=resolution // 5)
        plt.close(fig)
        #print(f"{view_name} saved at {file_path}")
        i+=1


def load_point_cloud_from_txt(filepath):
    """
    Load point cloud data from a .txt file.
    """
    return np.loadtxt(filepath)
    
def process_all_point_clouds(data_dir, save_dir, use_reflexity=False, resolution=512):
    #for root, dirs, files in os.walk(data_dir):
    for folder_name in [ "traffic_light", "traffic_sign", "person", "car", "bicycle", "cart", "dog", "fence", "motorcycle", "rider", "scooter", "sign", "table", "traffic_guardrail", "tree", "umbrella", "wall"]:
        sub_folder = os.path.join(data_dir, folder_name)
        files =os.listdir(sub_folder)
        for file in files:
            if file.endswith('.txt'):  # Assuming point cloud files are .txt
                name = os.path.splitext(file)[0]
                file_path = os.path.join(sub_folder, file)
                #print(f"Processing {file_path}...")
                point_cloud_data = load_point_cloud_from_txt(file_path)

                # Generate 8 views for the current point cloud
                plot_8_views(name, point_cloud_data, use_reflexity, save_dir, resolution)
        print(f"finish {folder_name}" )

if __name__ == '__main__':

    data_dir = 'creat_dataset/data/instance_pointcloud/instance_seg_c17_rgb_txt/'
    save_dir = 'creat_dataset/Cap3D/captioning_pipeline/example_material/Cap3D_imgs'

    process_all_point_clouds(data_dir, save_dir, use_reflexity=False, resolution=512)
