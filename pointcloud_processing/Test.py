import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d

def calculate_fpfh_feature(points):
    #创建open3d点云对象
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    # 半径（radius）：用于确定用于拟合法线的点云邻域的大小。较小的半径可能会导致法线估计不准确，而较大的半径可能会导致法线受到平均化，使其在曲面处不准确。

    # 最大近邻数（max_nn）：规定了用于拟合法线的最大邻居数量。这是在法线估计过程中控制邻域点的数量的另一种方式。

    radius_normal=0.01 #kdtree 参数，用于估计法线的半径
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,max_nn=30))
    #设置FPFH特征计算的半径
    radius_feature = 0.3
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh.data
#np.linalg.norm 函数来计算 fpfh_feature 中每个向量的范数（也就是向量的长度）。axis=1 表示对每个向量的第二个维度进行操作，keepdims=True 保持结果的维度和输入相同。
#将 fpfh_feature 中的每个向量都除以其对应的范数，这得到了归一化后的特征向量。
# def normalize_fpfh_feature(fpfh_feature):
#     # 将每一行的特征归一化
#     return (fpfh_feature.T / np.linalg.norm(fpfh_feature, axis=1)).T

#调整FPFH特征的维度
def adjust_fpfh_dimension(fpfh_feature, target_dimension):
    current_dimension = fpfh_feature.shape[1]

    if current_dimension < target_dimension:
        zero_padding = np.zeros((fpfh_feature.shape[0], target_dimension - current_dimension))
        return np.concatenate((fpfh_feature, zero_padding), axis=1)
    elif current_dimension > target_dimension:
        return fpfh_feature[:, :target_dimension]
    else:
        return fpfh_feature

# threshold = 5.0
target_dimension=60
point_cloud_1 = np.load("/home/zhenghaoxiang/Hiwi/Scene1/2023_06_30_17_23_54/_0_pc_fused.npy")
point_cloud_2 = np.load("/home/zhenghaoxiang/Hiwi/Scene1/2023_06_30_17_28_06/_0_pc_fused.npy")

semantics_1 = point_cloud_1[:, 6]
instances_1 = point_cloud_1[:, 7]

semantics_2 = point_cloud_2[:, 6]
instances_2 = point_cloud_2[:, 7]

unique_semantics = np.unique(semantics_1)
unique_semantics = unique_semantics[unique_semantics > 2]

matched_instances_2 = np.copy(instances_2)

for semantic_label in unique_semantics:
    unique_instances_1 = np.unique(instances_1[np.where(semantics_1 == semantic_label)[0]])
    unique_instances_2 = np.unique(instances_2[np.where(semantics_2 == semantic_label)[0]])

    for instance_2 in unique_instances_2:
        indices_2 = np.where((semantics_2 == semantic_label) & (instances_2 == instance_2))[0]
        points_2 = point_cloud_2[indices_2, :3]

        fpfh_2 = calculate_fpfh_feature(points_2)
        fpfh_2=adjust_fpfh_dimension(fpfh_2,target_dimension)

        min_distance = float('inf')
        nearest_instance_1 = None

        for instance_1 in unique_instances_1:
            indices_1 = np.where((semantics_1 == semantic_label) & (instances_1 == instance_1))[0]
            points_1 = point_cloud_1[indices_1, :3]

            fpfh_1 = calculate_fpfh_feature(points_1)
            fpfh_1 = adjust_fpfh_dimension(fpfh_1,target_dimension)

            # 使用欧氏距离计算两个特征之间的距离
            distance = np.linalg.norm(fpfh_1 - fpfh_2)

            if distance < min_distance:
                min_distance = distance
                nearest_instance_1 = instance_1

        matched_instances_2[indices_2] = nearest_instance_1
        print(matched_instances_2[indices_2])

if np.array_equal(matched_instances_2, instances_2):
    print("matched_instances_2 is the same as instances_2")
else:
    print("matched_instances_2 is different from instances_2")

point_cloud_2[:, 7] = matched_instances_2
np.save("updated_point_cloud_2.npy", point_cloud_2)

      

        
        


print("fertig")
print(instances_1)
print(instances_2)
print(matched_instances_2)