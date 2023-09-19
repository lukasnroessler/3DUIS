import hdbscan
import numpy as np
import open3d as o3d


def clusters_hdbscan(points_set):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=20, min_samples=None
                            )

    clusterer.fit(points_set)

    labels = clusterer.labels_.copy()

    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1
    print(set(labels))
    return labels

def clusters_from_pcd(pcd):
    # clusterize pcd points
    labels = np.array(pcd.cluster_dbscan(eps=0.25, min_points=10))
    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels

def clusterize_pcd(points, scan_path):
    scan_info = scan_path.split('/')
    scan_file = scan_info[-1].split('.')[0]
    seq_num = scan_info[-3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # instead of ransac use patchwork
    ground_file = f'./Datasets/SemanticKITTI/assets/patchwork/{seq_num}/{scan_file}.label'
    ground_labels = np.fromfile(ground_file, dtype=np.uint32)
    ground_labels.reshape((-1))
    inliers = list(np.where(ground_labels == 9)[0])

    pcd_ = pcd.select_by_index(inliers, invert=True)
    labels_ = np.expand_dims(clusters_hdbscan(np.asarray(pcd_.points)), axis=-1)

    labels = np.ones((points.shape[0], 1)) * -1
    mask = np.ones(labels.shape[0], dtype=bool)
    mask[inliers] = False
    
    labels[mask] = labels_

    return np.concatenate((points, labels), axis=-1)


def clusterize_anovox_pcd(pcd, sem_pcd): # inputs are o3d.geometry.PointCloud()
    # clusterizes background (road, sidewalk, road line)
    points = pcd.points
    labels = sem_pcd.colors

    ground = np.array([[128, 64, 128], # road 
                        [244, 35, 232], # sidewalk
                        [157, 234, 50], # road line
                        [81, 0, 81],])   # ground
    
    labels = (np.asarray(labels) * 255.0).astype(np.uint8)

    # sample out all points that represent ground
    inliers = []
    for index, label in enumerate(labels):
        if (label in ground):
            inliers.append(index)

    # inliers = list(np.where((labels not in ground)))
    # print("inliers ",inliers)
    pcd_ = pcd.select_by_index(inliers, invert=True)
    # print("pcd_:", pcd_)
    labels_ = np.expand_dims(clusters_hdbscan(np.asarray(pcd_.points)), axis=-1)
    labels = np.ones((np.asarray(points).shape[0], 1)) * -1
    mask = np.ones(labels.shape[0], dtype=bool)
    mask[inliers] = False
    
    cluster_indices = []
    for cluster_i, cluster in enumerate(labels_):
        if cluster == 20.0:
            cluster_indices.append(cluster_i)

    # sem_pcd.colors[cluster_indices] = o3d.utility.Vector3dVector(np.asarray([0.99, 0.99, 0]))
    new_pcd = sem_pcd.select_by_index(cluster_indices)
    # o3d.visualization.draw_geometries([new_pcd])    

    labels[mask] = labels_ 
    intensity = np.ones(labels.shape) * 250
    output_arr = np.concatenate((points, intensity, labels), axis=-1)
    return output_arr 

