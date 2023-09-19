import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os

from utils.pcd_preprocess import *


class AnoVoxDataLoader(Dataset):
    def __init__(self, root,  split='train'):
        self.root = root
        # self.augmented_dir = 'augmented_views_patchwork'

        # if not os.path.isdir(os.path.join(self.root, 'assets', self.augmented_dir)):
        #     os.makedirs(os.path.join(self.root, 'assets', self.augmented_dir))

        

        # self.seq_ids = {}
        # self.seq_ids['train'] = [ '00' , '01', '02', '03', '04', '05', '06', '07', '09', '10']
        # self.seq_ids['validation'] = ['08']
        # self.seq_ids['test'] = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        # self.split = split

        assert (split == 'train' or split == 'validation' or split == 'test')
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list()

        print('The size of %s data is %d'%(split,len(self.points_datapath)))

        self.COLOR_PALETTE = (
            np.array(
                [
                    (0, 0, 0),          # unlabeled     =   0u
                    (128, 64, 128),     # road          =   1u
                    (244, 35, 232),     # sidewalk      =   2u
                    (70, 70, 70),       # building      =   3u
                    (102, 102, 156),    # wall          =   4u
                    (190, 153, 153),    # fence         =   5u
                    (153, 153, 153),    # pole          =   6u
                    (250, 170, 30),     # traffic light =   7u
                    (220, 220, 0),      # traffic sign  =   8u
                    (107, 142, 35),     # vegetation    =   9u
                    (152, 251, 152),    # terrain       =  10u
                    (70, 130, 180),     # sky           =  11u
                    (220, 20, 60),      # pedestrian    =  12u
                    (255, 0, 0),        # rider         =  13u
                    (0, 0, 142),        # Car           =  14u
                    (0, 0, 70),         # truck         =  15u
                    (0, 60, 100),       # bus           =  16u
                    (0, 80, 100),       # train         =  17u
                    (0, 0, 230),        # motorcycle    =  18u
                    (119, 11, 32),      # bicycle       =  19u
                    (110, 190, 160),    # static        =  20u
                    (170, 120, 50),     # dynamic       =  21u
                    (55, 90, 80),       # other         =  22u
                    (45, 60, 150),      # water         =  23u
                    (157, 234, 50),     # road line     =  24u
                    (81, 0, 81),        # ground         = 25u
                    (150, 100, 100),    # bridge        =  26u
                    (230, 150, 140),    # rail track    =  27u
                    (180, 165, 180),    # guard rail    =  28u
                    (250, 128, 114),    # home          =  29u
                    (255, 36, 0),       # animal        =  30u
                    (224, 17, 95),      # nature        =  31u
                    (184, 15, 10),      # special       =  32u
                    (245, 0, 0),        # airplane      =  33u
                    (245, 0, 0),        # falling       =  34u
                ]
            )
        )  # normalize each channel [0-1] since this is what Open3D uses


    def datapath_list(self):
        print("root", self.root)
        self.points_datapath = []
        self.labels_datapath = []
        self.instance_datapath = []

        for scenario in os.listdir(self.root):
            if scenario == 'Scenario_Configuration_Files':
                continue
            point_dir = os.path.join(self.root, scenario, 'PCD')

            # print("point dir:", os.listdir(point_dir))
            # os.listdir(point_dir).sort()
            sem_point_dir = os.path.join(self.root, scenario, "SEMANTIC_PCD")
            # os.listdir(sem_point_dir).sort()
            try:
                self.points_datapath += [ os.path.join(point_dir, point_file) for point_file in os.listdir(point_dir)]
                print("points datapath:", self.points_datapath)
                self.labels_datapath += [ os.path.join(sem_point_dir, sem_point_file) for sem_point_file in os.listdir(sem_point_dir) ]
            except:
                pass
        
        # print("points datapath:", self.points_datapath)


        # for seq in self.seq_ids[split]:
        #     point_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'velodyne')
        #     point_seq_bin = os.listdir(point_seq_path)
        #     point_seq_bin.sort()
        #     self.points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]

        #     try:
        #         label_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'labels')
        #         point_seq_label = os.listdir(label_seq_path)
        #         point_seq_label.sort()
        #         self.labels_datapath += [ os.path.join(label_seq_path, label_file) for label_file in point_seq_label ]
        #     except:
        #         pass

        #     try:
        #         instance_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'instances')
        #         point_seq_instance = os.listdir(instance_seq_path)
        #         point_seq_instance.sort()
        #         self.instance_datapath += [ os.path.join(instance_seq_path, instance_file) for instance_file in point_seq_instance ]
        #     except:
        #         pass        

    def __getitem__(self, index):
        pcd = self.points_datapath[index]
        pcd = o3d.io.read_point_cloud(pcd)
        points_set = np.asarray(pcd.points)

        semantic_pcd = self.labels_datapath[index]
        semantic_pcd = o3d.io.read_point_cloud(semantic_pcd)
        # semantic_points = np.asarray(semantic_pcd.points)
        color_labels = np.asarray(semantic_pcd.colors)
        points_set = clusterize_anovox_pcd(pcd, semantic_pcd)
        # points_set = np.concatenate((points_set,np.ones((points_set.shape[0],1))), axis=1)
        # visualize_pcd_clusters(points_set, points_set, points_set, semantic_pcd)
        


        # transform color labels to labels as integer value
        sem_labels = (np.asarray(color_labels) * 255.0).astype(np.uint8)
        new_labels = np.arange(len(sem_labels))
        for i, value in enumerate(sem_labels): # 
            color_index = np.where((self.COLOR_PALETTE == value).all(axis = 1))
            # print(value)
            # for color in COLOR_PALETTE:
            #     if (value == color).all():                
            #         new_labels[i] = np.where((COLOR_PALETTE == value))[0][0]
            # print(color_index)
            new_labels[i] = color_index[0][0]
        sem_labels = new_labels

        return {'points_cluster': points_set, 'semantic_label': sem_labels, 'scan_file': self.points_datapath[index]}
        # return {'points_cluster': points_set, 'scan_file': self.points_datapath[index]}

    
    def __len__(self):
        return len(self.points_datapath)
            

def visualize_pcd_clusters(p, p_corr, p_slc, gt, cmap="viridis", center_point=None, quantize=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p[:,:3])

    labels = p[:, -1]
    colors = plt.get_cmap(cmap)(labels)

    # labels = p_slc[:, -1][:,np.newaxis]
    # colors = np.concatenate((labels, labels, labels), axis=-1)
    
    # if center_point is not None:
    #     lbl = np.argsort(labels)
    #     colors[lbl[-20:],:3] = [1., 0., 0.]
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    pcd_corr = o3d.geometry.PointCloud()
    pcd_corr.points = o3d.utility.Vector3dVector(p_corr[:,:3])

    labels = p_corr[:, -1]
    colors = plt.get_cmap(cmap)(labels)
    
    if center_point is not None:
        lbl = np.argsort(labels)
        colors[lbl[-20:],:3] = [1., 0., 0.]
    pcd_corr.colors = o3d.utility.Vector3dVector(colors[:, :3])

    pcd_slc = o3d.geometry.PointCloud()
    pcd_slc.points = o3d.utility.Vector3dVector(p_slc[:,:3])

    labels = p_slc[:, -1]
    colors = plt.get_cmap(cmap)(labels)
    
    if center_point is not None:
        lbl = np.argsort(labels)
        colors[lbl[-20:],:3] = [1., 0., 0.]
    pcd_slc.colors = o3d.utility.Vector3dVector(colors[:, :3])

    pcd = pcd.voxel_down_sample(voxel_size=5)
    pcd_corr = pcd_corr.voxel_down_sample(voxel_size=5)
    pcd_slc = pcd_slc.voxel_down_sample(voxel_size=5)
    gt = gt.voxel_down_sample(voxel_size=5)

    colors_gt = np.asarray(gt.colors).copy()
    colors_gt[:,0] = np.maximum(colors_gt[:,0], 0.2)
    colors = np.asarray(pcd.colors)

    colors[:,0] = colors_gt[:,0]*colors[:,0]
    colors[:,1] = colors_gt[:,0]*colors[:,1]
    colors[:,2] = colors_gt[:,0]*colors[:,2]
    #colors = plt.get_cmap(cmap)(colors)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd, pcd_corr, pcd_slc, gt])