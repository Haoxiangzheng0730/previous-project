import time
import open3d as o3d
import copy
import numpy as np


voxel_size = 0.1
# 传入点云数据 计算FPFH 快速点特征直方图(Fast Point Feature Histograms, FPFH)是PFH计算方式的简化形式。
# 它的思想在于分别计算查询点的k邻域中每一个点的简化点特征直方图(Simplified Point Feature Histogram,SPFH)，
# 再通过一个公式将所有的SPFH加权成最后的快速点特征直方图。FPFH把算法的计算复杂度降低到了O(nk) ，但是任然保留了PFH大部分的识别特性。
# ————————————————
def FPFH_Compute(pcd):

    radius_normal = voxel_size * 2 #kdtree参数，用于估计法线的半径，
    print("::Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,max_nn=30))
# 估计法线的一个参数，使用混合型的Kdtree，半径内取最多30个邻居
    radius_feature=voxel_size * 5 #kdtree 参数，用于估计FPFH特征的半径
    print("::Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh=o3d.pipelines.registration.compute_fpfh_feature(pcd,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,max_nn=100))# 计算FPFH特征，搜索方法kdtree
    return pcd_fpfh
# RANSAC 配准
def execute_global_registration(source,target,source_fpfh,target_fpfh):#传入两个点云和点云特征
    distance_threshold = voxel_size * 0.5 # 设定距离阈值
    print("we use a liberal distance threshold %.3f." % distance_threshold)
#2个点云，两个点云的特征，距离阈值，一个函数：4,
#一个list[0.9的两个对应点的线段长度阈值，两个点的距离阈值]，
#一个函数设定最大迭代次数和最大验证次数
    result=o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,target,source_fpfh,target_fpfh,True,distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],o3d.pipelines.registration.RANSACConvergenceCriteria(10000,0.99)
    )
    return result
def draw_registration_result(source,target,window_name,transformation):
    source_temp=copy.deepcopy(source)
    target_temp=copy.deepcopy(target)
    source_temp.paint_uniform_color([1,0,0])
    target_temp.paint_uniform_color([0,1,0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp,target_temp],window_name,width=600,height=600)
def find_overlapped_cloud(cloud1, cloud2):
    overlapped_cloud_indices = []
    octree = o3d.geometry.Octree(max_depth=4)
    octree.convert_from_point_cloud(cloud1, size_expand=0.01)
    min_pt = octree.get_min_bound()
    max_pt = octree.get_max_bound()
    for point in cloud2.points:
        if point[0] < min_pt[0] or point[1] < min_pt[1] or point[2] < min_pt[2] or \
                    point[0] > max_pt[0] or point[1] > max_pt[1] or point[2] > max_pt[2]:
            continue
        else:
            leaf_node, leaf_info = octree.locate_leaf_node(point)
            if leaf_info is not None:
                indices = leaf_node.indices
                for indice in indices:
                    overlapped_cloud_indices.append(indice)

    return cloud1.select_by_index(overlapped_cloud_indices)

def find_different_cloud(cloud1, cloud2):
    overlapped_cloud_indices = []
    pcl_diff = o3d.geometry.PointCloud()
    octree = o3d.geometry.Octree(max_depth=4)
    octree.convert_from_point_cloud(cloud1, size_expand=0.01)
    min_pt = octree.get_min_bound()
    max_pt = octree.get_max_bound()
    for point in cloud2.points:
        if point[0] < min_pt[0] or point[1] < min_pt[1] or point[2] < min_pt[2] or \
                    point[0] > max_pt[0] or point[1] > max_pt[1] or point[2] > max_pt[2]:
            continue
        else:
            leaf_node, leaf_info = octree.locate_leaf_node(point)
            if leaf_info is None:
                pcl_diff.points.append(point)

    return pcl_diff

def display_inlier_outlier(cloud, m_ind):
    inlier_cloud = cloud.select_by_index(m_ind)
    outlier_cloud = cloud.select_by_index(m_ind, invert=True)

    print("showing non overla(red) and overlap(green):")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                          window_name="重叠和非重叠点",
                                          left=50, top=50,
                                          mesh_show_back_face=False)

if __name__ == "__main__":
    #读取点云

    source_original = o3d.io.read_point_cloud("scene0050_00_vh_clean_2.labels.ply")
    target_original = o3d.io.read_point_cloud("scene0050_01_vh_clean_2.labels.ply")
    print(target_original)
    source = source_original.voxel_down_sample(voxel_size)
    target = target_original.voxel_down_sample(voxel_size)
    source.paint_uniform_color([1,0,0])
    target.paint_uniform_color([0,1,0])
    o3d.visualization.draw_geometries([source,target],window_name="原始点云与目标点云",width=600,height=600)
    source_fpfh=FPFH_Compute(source)
    target_fpfh=FPFH_Compute(target)
    #调用RANSAC执行配准
    start=time.time()
    result_ransac=execute_global_registration(source,target,source_fpfh,target_fpfh)
    print("Global registration took %.3f sec.\n" %(time.time()-start))
    print(result_ransac)#输出RANSAC配准信息
    Tr=result_ransac.transformation
    draw_registration_result(source,target,"RANSAC粗配准",Tr)#可视化配准结果
    #ICP配准
    start=time.time()
    icp_p2plane=o3d.pipelines.registration.registration_icp(
        source,target,0.5,result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),#执行点对面的ICP算法
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    )
    print("ICP registration took %.3f sec. \n"%(time.time()-start))
    print(icp_p2plane)
    print("Transformation is:")
    print(icp_p2plane.transformation)# 输出变换矩阵
    draw_registration_result(source_original,target_original,"ICP精配准",icp_p2plane.transformation)


#显示重叠部分
    cloud1 =source_original
    cloud1.transform(icp_p2plane.transformation)
    cloud2 =target_original
    ##overlapped_cloud1 = find_overlapped_cloud(cloud2, cloud1)
    ##overlapped_cloud2 = find_overlapped_cloud(cloud1, cloud2)
    ##o3d.visualization.draw_geometries([overlapped_cloud2],window_name="重叠部分",
    ##                                  width=1024, height=768,
    ##                                  left=50, top=50,
    ##                                  mesh_show_back_face=False)


# show the different points
    start = time.time()
    diff_12 = find_different_cloud(cloud1, cloud2)
    diff_21 = find_different_cloud(cloud2, cloud1)
    end = time.time()
    print(end - start)
    o3d.visualization.draw_geometries([diff_12], window_name="12",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)
    o3d.visualization.draw_geometries([diff_21], window_name="21",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)