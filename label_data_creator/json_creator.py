#!/usr/bin/env python
#python常见开头，表示这是一个脚本
import json
from nturl2path import pathname2url
from PIL import Image as ImageSaver
import os
import tiledb
import numpy as np



class JsonCreator:

    def __init__(self):
        self.c_cam_fx = 695.3909
        self.c_cam_fy = 744.3789
        self.c_cam_cx = 728.9554
        self.c_cam_cy = 547.7995
        self.c_cam_timestamp = 0

        self.l_cam_fx = 693.418
        self.l_cam_fy = 742.785
        self.l_cam_cx = 721.044
        self.l_cam_cy = 597.6945
        self.l_cam_timestamp = 0

        self.r_cam_fx = 694.527
        self.r_cam_fy = 742.256
        self.r_cam_cx = 752.435
        self.r_cam_cy = 548.953
        self.r_cam_timestamp = 0
      
        
        ##设置了一些相机参数的默认值，包括中心相机(c_cam)、左侧相机(l_cam)
        # 和右侧相机(r_cam)的焦距(fx和fy)、光心(cx和cy)和时间戳(timestamp)。
    
    

    def set_path(self, path):
        self.path = path
    
        ##set_path方法用于设置保存JSON文件的路径。通过调用这个方法，可以指定JSON文件保存在哪个目录下。
    def set_scene_name(self,scene_name):
        self.scene_name=scene_name



    # def set_scene_name(self,scene_index):
    #     self.scene_name=f"scene_{scene_index}"
    #     scene_folder_path=os.path.join(self.path,self.scene_name)
    #     os.makedirs(scene_folder_path,exist_ok=True)
    #     self.scene_folder_path=scene_folder_path

    def only_pointcloud(self, set):
        self.only_pointcloud = set
        ##only_pointcloud方法用于设置是否只保存点云数据而不保存相机图像数据。可以通过调用这个方法来决定是否保存相机图像数据。

    def set_pointcloud_xyz(self, pcd, timestamp, position, quaternion):
        self.pointcloud = []
        for i in range(len(pcd['x'])):
            self.pointcloud.append({'x': float((pcd['x'][i])), 'y': float(pcd['y'][i]), 'z': float(pcd['z'][i])})
        self.pcd_timestamp = timestamp
        self.lidar_position = {'x': position[0], 'y': position[1], 'z': position[2]}
        self.lidar_heading = {'x': quaternion[0], 'y': quaternion[1], 'z': quaternion[2], 'w': quaternion[3]}

    def set_pointcloud_xyzi(self, pcd, timestamp, position, quaternion):
        xyzi_dtype = np.dtype([('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'),('i', 'i1')])
        self.pointcloud = np.empty(len(pcd['X']),dtype=xyzi_dtype)
        self.pointcloud = []
        for i in range(len(pcd['X'])):
            # self.pointcloud[i]['X'] = np.float64(pcd['X'][i])
            # self.pointcloud[i]['Y'] = np.float64(pcd['Y'][i])
            # self.pointcloud[i]['Z'] = np.float64(pcd['Z'][i])
            # self.pointcloud[i]['i'] = np.float32(pcd['Intensity'][i] / 255.0)
            self.pointcloud.append({'X': np.float64((pcd['X'][i])), 'Y': np.float64(pcd['Y'][i]), 'Z': np.float64(pcd['Z'][i]),
                                    'i': np.float64(pcd['Intensity'][i] / 255.)})
        self.pcd_timestamp = timestamp
        self.lidar_position = {'x': np.float64(position[0]), 'y': np.float64(position[1]), 'z':np.float64(position[2])}
        self.lidar_heading = {'x': np.float64(quaternion[0]), 'y':np.float64( quaternion[1]), 'z': np.float64(quaternion[2]), 'w': np.float64(quaternion[3])}

    def set_pointcloud_xyzrgb(self, pcd,rgb, timestamp, position, heading):

        self.pointcloud = []
        for i in range(len(pcd['x'])):
            self.pointcloud.append({'x': float((pcd['x'][i])), 'y': float(pcd['y'][i]), 'z': float(pcd['z'][i]),
                                    'r': int((pcd['r'][i])), 'g': int(pcd['g'][i]), 'b': int(pcd['b'][i])})
        self.pcd_timestamp = timestamp
        self.lidar_position = {'x': position[0], 'y': position[1], 'z': position[2]}
        self.lidar_heading = {'x': heading[0], 'y': heading[1], 'z': heading[2], 'w': heading[3]}

    def set_center_image(self, image, image_url, timestamp, position, quaternion):
        self.c_cam_heading = {'x': np.float64(quaternion[0]), 'y': np.float64(quaternion[1]), 'z':np.float64( quaternion[2]), 'w': np.float64(quaternion[3])}
        self.c_cam_position = {'x': np.float64(position[0]), 'y': np.float64(position[1]), 'z': np.float64(position[2])}
        self.c_cam_image_url = image_url
        self.c_cam_timestamp = timestamp
        im = ImageSaver.fromarray(image)
        im.save(self.path+image_url)
        

    def set_left_image(self, image, image_url, timestamp, position, quaternion):
        self.l_cam_heading = {'x': np.float64(quaternion[0]), 'y': np.float64(quaternion[1]), 'z':np.float64( quaternion[2]), 'w': np.float64(quaternion[3])}
        self.l_cam_position = {'x': np.float64(position[0]), 'y': np.float64(position[1]), 'z': np.float64(position[2])}
        self.l_cam_image_url = image_url
        self.l_cam_timestamp = timestamp
        im = ImageSaver.fromarray(image)
        im.save(self.path+image_url)
        

    def set_right_image(self, image, image_url, timestamp, position, quaternion):
        self.r_cam_heading = {'x': np.float64(quaternion[0]), 'y': np.float64(quaternion[1]), 'z':np.float64( quaternion[2]), 'w': np.float64(quaternion[3])}
        self.r_cam_position = {'x': np.float64(position[0]), 'y': np.float64(position[1]), 'z': np.float64(position[2])}
        self.r_cam_image_url = image_url
        self.r_cam_timestamp = timestamp
        im = ImageSaver.fromarray(image)
        im.save(self.path+image_url)
        


    def save(self, filename):
        images = []
        if not self.only_pointcloud:
            image_c = {'fx': self.c_cam_fx, 'fy': self.c_cam_fy, 'cx': self.c_cam_cx, 'cy': self.c_cam_cy,
                       'p1': 0, 'p2': 0, 'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0,
                       'image_url': self.c_cam_image_url,
                       'timestamp': self.c_cam_timestamp,
                       'position': self.c_cam_position,
                       'heading': self.c_cam_heading,
                       'camera_model': 'pin_hole'}

            image_l = {'fx': self.l_cam_fx, 'fy': self.l_cam_fy, 'cx': self.l_cam_cx, 'cy': self.l_cam_cy,
                       'p1': 0, 'p2': 0, 'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0,
                       'image_url': self.l_cam_image_url,
                       'timestamp': self.l_cam_timestamp,
                       'position': self.l_cam_position,
                       'heading': self.l_cam_heading,
                       'camera_model': 'pin_hole'}

            image_r = {'fx': self.r_cam_fx, 'fy': self.r_cam_fy, 'cx': self.r_cam_cx, 'cy': self.r_cam_cy,
                       'p1': 0, 'p2': 0, 'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0,
                       'image_url': self.r_cam_image_url,
                       'timestamp': self.r_cam_timestamp,
                       'position': self.r_cam_position,
                       'heading': self.r_cam_heading,
                       'camera_model': 'pin_hole'}

            images.append(image_c)
            images.append(image_l)
            images.append(image_r)

        data = {"images": images, 'device_position': self.lidar_position, 'device_heading': self.lidar_heading,
                    'timestamp': self.pcd_timestamp, "points": self.pointcloud}
        
        with open(self.path + filename + '.json', 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '))

        print('saved as ' + filename + '.json')
            
       



