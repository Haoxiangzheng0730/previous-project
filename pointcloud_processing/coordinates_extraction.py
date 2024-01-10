import os
import json
import numpy as np
import open3d as o3d
from plyfile import PlyData


def extract_coordinates(ply_path, label_data):
    ply = PlyData.read(ply_path)
    plydat = ply['vertex'].data
    plydat = np.array(plydat)

    updated_label_data = label_data.copy()

    seggroups = label_data['segGroups']
    for seg in seggroups:
        obj = seg['label']
        index = seg['segments']
        id = seg['id']

        obj_ply = plydat[index]
        obj_ply_data = []
        for p in obj_ply:
            x, y, z, r, g, b = p[:6]
            instance = id
            obj_ply_data.append([x, y, z, r, g, b, instance])
        obj_ply_data = np.array(obj_ply_data)

        updated_label_data['segGroups'][obj]['segments_xyz'] = obj_ply_data[:, :3].tolist()

    return updated_label_data


def update_point_cloud(pcd1, pcd2, label_data1, label_data2):
    updated_pcd1 = pcd1.copy()

    for seg2 in label_data2['segGroups']:
        obj2 = seg2['label']
        if obj2 in label_data1['segGroups']:
            seg1 = label_data1['segGroups'][obj2]
            obj_pcd2 = np.array(seg2['segments_xyz'])
            obj_pcd1 = np.array(seg1['segments_xyz'])

            for p2 in obj_pcd2:
                distances = np.linalg.norm(obj_pcd1 - p2, axis=1)
                min_index = np.argmin(distances)
                min_distance = distances[min_index]

                if min_distance < 0.01:  # Set a threshold for movement detection
                    # Object moved, update coordinates in pcd1
                    updated_pcd1[seg1['segments'][min_index]] = p2
                else:
                    # Object disappeared, remove coordinates from pcd1
                    updated_pcd1 = np.delete(updated_pcd1, seg1['segments'][min_index], axis=0)
                    seg1['segments'].remove(seg1['segments'][min_index])

        else:
            # New object in pcd2, update pcd1 and add to corresponding label
            obj_pcd2 = np.array(seg2['segments_xyz'])
            new_segments = []

            for p2 in obj_pcd2:
                updated_pcd1 = np.vstack((updated_pcd1, p2))
                new_segment_index = updated_pcd1.shape[0] - 1
                seg2['segments'].append(new_segment_index)
                new_segments.append(new_segment_index)

            seg1 = seg2.copy()
            seg1['segments'] = new_segments
            label_data1['segGroups'][obj2] = seg1

    return updated_pcd1, label_data1


# 输入点云1的ply和json文件路径
ply_path1 = 'scene0000_01_vh_clean_2.labels.ply'
label_path1 = '.json'

# 输入点云2的ply和json文件路径
ply_path2 = 'path/to/pointcloud2.ply'
label_path2 = 'path/to/label2.json'

# 读取点云1和标签1
ply1 = o3d.io.read_point_cloud(ply_path1)
label_file1 = open(label_path1, 'r')
label_data1 = json.load(label_file1)

# 读取点云2和标签2
ply2 = o3d.io.read_point_cloud(ply_path2)
label_file2 = open(label_path2, 'r')
label_data2 = json.load(label_file2)

# 提取点云1中的坐标
pcd1 = np.asarray(ply1.points)

# 提取点云1中的坐标并更新标签数据
updated_label_data1 = extract_coordinates(ply_path1, label_data1)

# 更新点云1和标签1
updated_pcd1, updated_label_data1 = update_point_cloud(pcd1, np.asarray(ply2.points), updated_label_data1, label_data2)

# 将更新后的点云1保存为ply文件
output_ply_path = 'path/to/updated_pointcloud1.ply'
o3d.io.write_point_cloud(output_ply_path, o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(updated_pcd1)))

# 将更新后的标签1保存为json文件
output_label_path = 'path/to/updated_label1.json'
with open(output_label_path, 'w') as output_label_file:
    json.dump(updated_label_data1, output_label_file)

# 输出更新后的点云1和标签1的路径
print("Updated point cloud 1 saved as:", output_ply_path)
print("Updated label 1 saved as:", output_label_path)