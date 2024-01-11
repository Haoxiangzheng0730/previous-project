import rospy
from sensor_msgs.msg import PointCloud2
# import ros_numpy
# from json_creator import JsonCreator
import os
import numpy as np
# import open3d as o3d

import string

import json

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs.point_cloud2 import create_cloud 
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
    


rospy.init_node('json_pointcloud_publisher')
cloud_publisher = rospy.Publisher('/cloud', PointCloud2, queue_size=10, latch=True)
lidar_poses_publisher = rospy.Publisher('/lidars', PoseArray, queue_size=10, latch=True)
camera_l_poses_publisher = rospy.Publisher('/l_cameras', PoseArray, queue_size=10, latch=True)

fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('intensity', 12, PointField.FLOAT32, 1)
          ]

# fields = [PointField('x', 0, PointField.FLOAT32, 1),
#           PointField('y', 4, PointField.FLOAT32, 1),
#           PointField('z', 8, PointField.FLOAT32, 1)
#           ]
header = Header()
header.frame_id = "wheel_odom"



# PATH = "/home/hao/IFLProject/aribic/2_IFL_code/das_reader/json_data/"
PATH = "/home/zhenghaoxiang/Hiwi/json/"
# PATH = "/home/hao/IFLProject/aribic/2_IFL_code/pilot_Data/sequence_frames_intensity_1/"
# PATH = "/home/hao/IFLProject/aribic/2_IFL_code/pilot_Data/sequence_frames_intensity_2/"
json_files = [] 
for file in os.listdir(PATH):
    if file.find(".json") >0:
        json_files.append(PATH + file)



rate = rospy.Rate(1)

points_pc2 = []
it = []
lidars = PoseArray()
lidars.header.frame_id = "wheel_odom"

l_cameras = PoseArray()
l_cameras.header.frame_id = "wheel_odom"

for file in json_files:
    print("read ",file)
    with open(file, 'r') as f:
        points = json.load(f)
        # print(points['images'][0]['position'])
        # print(points['images'][0]['heading'])
        # print(points['images'][1]['position'])
        # print(points['images'][1]['heading'])
        # print(points['images'][2]['position'])
        # print(points['images'][2]['heading'])        
        # print(points['device_position'])
        # print(points['device_heading'])
        # print("-------")
        tmp_pose = Pose()
        tmp_pose.position.x = points['device_position']['x']
        tmp_pose.position.y = points['device_position']['y']
        tmp_pose.position.z = points['device_position']['z']
        
        tmp_pose.orientation.x = points['device_heading']['x']
        tmp_pose.orientation.y = points['device_heading']['y']
        tmp_pose.orientation.z = points['device_heading']['z']
        tmp_pose.orientation.w = points['device_heading']['w']


        lidars.poses.append(tmp_pose)

        tmp_pose_cam0 = Pose()
        tmp_pose_cam0.position.x = points['images'][0]['position']['x']
        tmp_pose_cam0.position.y = points['images'][0]['position']['y']
        tmp_pose_cam0.position.z = points['images'][0]['position']['z']
        
        tmp_pose_cam0.orientation.x = points['images'][0]['heading']['x']
        tmp_pose_cam0.orientation.y = points['images'][0]['heading']['y']
        tmp_pose_cam0.orientation.z = points['images'][0]['heading']['z']
        tmp_pose_cam0.orientation.w = points['images'][0]['heading']['w']

        l_cameras.poses.append(tmp_pose_cam0)


        for point in points['points']:
            points_pc2.append([point['X'], point['Y'], point['Z'], point['i']])
        # points_array = np.array(points['points'][0]['x'])
        # print(type(points_array))
        # print(points_array['x'])
        print(len(points_pc2))
       
pc2 = create_cloud(header, fields, points_pc2)
cloud_publisher.publish(pc2)
lidar_poses_publisher.publish(lidars)
camera_l_poses_publisher.publish(l_cameras)
print("published")

# print(ros_numpy.numpify(pc2))
    # rate.sleep()

# print("received a map")
#     # print(data.fields)
#     # print(data.point_step)
#     # print(data.row_step)
#     # print(len(data.data))
#     pc = ros_numpy.numpify(data)
#     pc = ros_numpy.point_cloud2.split_rgb_field(pc)





# rospy.spin()