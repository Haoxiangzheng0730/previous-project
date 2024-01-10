import numpy as np
from scipy.spatial import KDTree
import open3d as o3d


def find_matching_boxes(bounding_boxes_cloud1, bounding_boxes_cloud2, threshold):
    # 提取第一幅点云中bounding box的中心点
    center_points_cloud1 = np.array([(box[0] + box[3]) / 2, (box[1] + box[4]) / 2, (box[2] + box[5]) / 2)
                                    for box in bounding_boxes_cloud1.values()])

    # 提取第二幅点云中bounding box的中心点
    center_points_cloud2 = np.array([(box[0] + box[3]) / 2, (box[1] + box[4]) / 2, (box[2] + box[5]) / 2)
                                    for box in bounding_boxes_cloud2.values()])

    # 创建两个k-d树
    kdtree_cloud1 = KDTree(center_points_cloud1)
    kdtree_cloud2 = KDTree(center_points_cloud2)

    # 用k-d树在第二幅点云中找到第一幅点云每个bounding box的中心点的最近邻居
    _, indices = kdtree_cloud2.query(center_points_cloud1, k=1)

    # 计算距离，并标记匹配和不同的bounding box
    matching_indices = []
    different_indices = []
    for i, idx in enumerate(indices):
        distance = np.linalg.norm(center_points_cloud1[i] - center_points_cloud2[idx])
        if distance < threshold:
            matching_indices.append(i)
        else:
            different_indices.append(i)

    return matching_indices, different_indices


def visualize_difference(point_cloud1, point_cloud2, matching_indices, different_indices):
    # 将点从npy转换为Open3D PointCloud
    def convert_to_pointcloud(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)  # 将RGB值归一化
        return pcd



    # 获取有差别的bounding box中的点
    different_points_cloud1 = np.concatenate([point_cloud1[point_cloud1[:, -1] == instance][:, :3]
                                              for i, instance in enumerate(instance_cloud1) if i in different_indices], axis=0)
    different_points_cloud2 = np.concatenate([point_cloud2[point_cloud2[:, -1] == instance][:, :3]
                                              for i, instance in enumerate(instance_cloud2) if i in different_indices], axis=0)

    # 转换为Open3D PointCloud
    pcd1 = convert_to_pointcloud(different_points_cloud1)
    pcd2 = convert_to_pointcloud(different_points_cloud2)

    # 将点云1的颜色设为红色
    pcd1.paint_uniform_color([1, 0, 0])
    # 将点云2的颜色设为蓝色
    pcd2.paint_uniform_color([0, 0, 1])

    # 将两个点云合并以进行可视化
    combined_pcd = pcd1 + pcd2

    # 可视化合并后的点云
    o3d.visualization.draw_geometries([combined_pcd])

point_cloud1=np.load("Scene1/2023_06_30_17_23_54/_0_pc_fused.npy")
point_cloud2=np.load("Scene1/2023_06_30_17_28_06/_0_pc_fused.npy")
#提取每个物体的boundingbox
bounding_boxes_cloud1={}
bounding_boxes_cloud2={}

instance_cloud1=np.unique(point_cloud1[:,-1])
instance_cloud2=np.unique(point_cloud2[:,-1])
for instance in instance_cloud1:
    points_instance_cloud1 = point_cloud1[point_cloud1[:, -1] == instance][:, :3]
    xmin, ymin, zmin = np.min(points_instance_cloud1, axis=0)
    xmax, ymax, zmax = np.max(points_instance_cloud1, axis=0)
    bounding_boxes_cloud1[instance] = [xmin, ymin, zmin, xmax, ymax, zmax]
for instance in instance_cloud2:
    points_instance_cloud2 = point_cloud2[point_cloud2[:, -1] == instance][:, :3]
    xmin, ymin, zmin = np.min(points_instance_cloud2, axis=0)
    xmax, ymax, zmax = np.max(points_instance_cloud2, axis=0)
    bounding_boxes_cloud2[instance] = [xmin, ymin, zmin, xmax, ymax, zmax]
matching_indices, different_indices = find_matching_boxes( bounding_boxes_cloud1, bounding_boxes_cloud2, threshold=0.01)

# 可视化有差异的bounding box中的点
visualize_difference(point_cloud1, point_cloud2, matching_indices, different_indices)