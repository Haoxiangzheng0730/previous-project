#!/usr/bin/env python
import json
from nturl2path import pathname2url
from PIL import Image as ImageSaver
import os
import tiledb
import numpy as np
from json_creator import JsonCreator
import transforms3d


'''
        TO DO
        transform the lidar pose into camera pose


[position,quaternion ] = transform(position_in, quaternion_in, trans, rotation)  !! rotation is quaternion
'''
def transform(position_in, quaternion_in, trans, rotation):
    # Convert quaternion to rotation matrix
    rotation_matrix = transforms3d.quaternions.quat2mat(quaternion_in)
    
    # Apply the given translation
    translated_position = position_in + trans
    
    # Apply the given rotation
    rotated_position = np.dot(rotation_matrix, translated_position)
    rotated_quaternion = transforms3d.quaternions.qmult(rotation, quaternion_in)
    
    return rotated_position, rotated_quaternion

def main():
# 设置场景文件夹路径
    base_folder = "/media/zhenghaoxiang/3CCCCE3ACCCDEDE8/20220815_160708_P2/"
    target_path="/home/zhenghaoxiang/Hiwi/json/"


    json_creator = JsonCreator()
    json_creator.only_pointcloud(False)  # 设置是否保存相机图像数据
    target_scene_folder=os.path.join(target_path)
# 遍历每个场景
    json_creator.set_path(target_scene_folder)
    for scene_index in range(1):
        os.makedirs(target_scene_folder,exist_ok=True)
        # json_creator.set_scene_name(f"scene_{scene_index}")
        scene_folder = os.path.join(base_folder, str(scene_index))

        pointcloud_folder = os.path.join(scene_folder, "pointcloud")
        l_rgb_folder = os.path.join(scene_folder,"l_rgb")
        r_rgb_folder = os.path.join(scene_folder, "r_rgb")
        c_rgb_folder = os.path.join(scene_folder, "c_rgb")
        # l_image_url = os.path.join(str(scene_index),"l_rgb.jpeg")
        # r_image_url = os.path.join(str(scene_index),"r_rgb.jpeg")
        # c_image_url = os.path.join(str(scene_index),"c_rgb.jpeg")
        l_image_url = str(scene_index)+"l_rgb.jpeg"
        r_image_url = str(scene_index)+"r_rgb.jpeg"
        c_image_url = str(scene_index)+"c_rgb.jpeg"
    
        odom_folder = os.path.join(scene_folder,"odom")
        odom_6DoF_folder = os.path.join(scene_folder,"odom_6DoF")
    


        timestamp = 0

        # 读取tiledb数据
        with tiledb.open(pointcloud_folder, mode="r") as array:
            pcd = array[:]
            
        with tiledb.open(odom_6DoF_folder,mode="r") as data:
            # data.dump()
        
            data = data[:]
            position =[data['x'][0],data['y'][0],data['z'][0]]
            quaternion = [data['q_x'][0], data['q_y'][0],data['q_z'][0],data['q_w'][0]]
        
            
        
        with tiledb.DenseArray(l_rgb_folder,mode="r") as image:
            l_image= image[:]
            r = np.array(l_image['r'])
            g = np.array(l_image['g'])
            b = np.array(l_image['b'])

            # 将各个列组合成numpy数组
            l_image = np.stack((r, g, b),axis=-1)
        with tiledb.DenseArray(r_rgb_folder,mode="r") as image:
            r_image= image[:]
            r = np.array(r_image['r'])
            g = np.array(r_image['g'])
            b = np.array(r_image['b'])

            # 将各个列组合成numpy数组
            r_image = np.stack((r, g, b),axis=-1)
        with tiledb.DenseArray(c_rgb_folder,mode="r") as image:
            c_image= image[:]
            r = np.array(c_image['r'])
            g = np.array(c_image['g'])
            b = np.array(c_image['b'])

            # 将各个列组合成numpy数组
            c_image = np.stack((r, g, b),axis=-1)

            

        json_creator.set_left_image(l_image,l_image_url,timestamp,position,quaternion)
        json_creator.set_right_image(r_image,r_image_url,timestamp,position,quaternion)
        json_creator.set_center_image(c_image,c_image_url,timestamp,position,quaternion)

        position_in = position
        quaternion_in = quaternion  # Example quaternion
        trans = np.array([0.1, 0.2, 0.3])
        #convert euler angle to quaternion
        rotation = transforms3d.euler.euler2quat(np.radians(30), np.radians(45), np.radians(60))

        position, quaternion = transform(position_in, quaternion_in, trans, rotation)
        print("Transformed Position:", position)
        print("Transformed Quaternion:", quaternion)

        # 创建jsoncreator对象并设置数据
        
        json_creator.set_pointcloud_xyzi(pcd, timestamp, position, quaternion)  # 设置点云数据
        


        #读取并设置相机图像数据
        # image_folder = os.path.join(scene_folder, "c_rgb")
        # for image_file in os.listdir(image_folder):
        #     image_path = os.path.join(image_folder, image_file)
        #     image = imagesaver.open(image_path)
        #     image_url = pathname2url(image_path)
        #     timestamp = 0
        #     position = [0, 0, 0]
        #     heading = [0, 0, 0, 0]
        #     json_creator.set_center_image(image, image_url, timestamp, position, heading)

        # 保存json文件
        json_creator.save(f"scene_{scene_index}")
if __name__== "__main__":
    main()