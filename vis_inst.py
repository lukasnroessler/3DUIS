import open3d as o3d
import numpy as np
from utils.corr_utils import crop_region
import matplotlib.pyplot as plt

pred = np.load('./output/3DUIS/Scenario_e9c61bb8-fe7e-4fde-8122-01e2181c0a62/raw_pred/PCD_748.npy')
# pred = np.load('./output/3DUIS/08/raw_pred/000000.npy')
# pred = np.load('./output/3DUIS/Scenario_f9903df8-9ca5-43c0-bf74-a3e5c1a72c11/raw_pred/PCD_156.npy')


points = pred[:,:3]
pred = pred[:,-1]  
print("size:", pred.size)
print("size of npy where 0", (np.where(pred == 0)[0]).size) 

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

colors_pred = np.zeros((len(pred), 4))
colors_cluster = np.zeros((len(pred), 4))
flat_indices = np.unique(pred)
max_instance = len(flat_indices)
colors_instance = plt.get_cmap("prism")(np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))

for idx in range(len(flat_indices)):
    colors_pred[pred == idx] = colors_instance[int(idx)]

colors_pred[pred == 0] = [0.,0.,0.,0.]

pcd.colors = o3d.utility.Vector3dVector(colors_pred[:,:3])

o3d.visualization.draw_geometries([pcd])
