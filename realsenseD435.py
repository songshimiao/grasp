#!/usr/bin/env python

import socket
import numpy as np
import cv2
import os
import time
import struct
import pyrealsense2 as rs

class RealsenseD435(object):

    def __init__(self):
        self.im_height = 720
        self.im_width = 1280
        self.intrinsics = None
        self.pipeline = None                                                                                                                                                                                                                        
        self.config = None
        self.scale = 0.001
        self.intrinsics = None
        
    def init_cam(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, 30)
        profile = self.pipeline.start(self.config)
        # Determine intrinsics
        rgb_profile = profile.get_stream(rs.stream.color)
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        self.intrinsics = np.array([raw_intrinsics.fx, 0, raw_intrinsics.ppx, 0, raw_intrinsics.fy, raw_intrinsics.ppy, 0, 0, 1]).reshape(3, 3)
        # self.intrinsics = np.array([607.879,0,325.14,0,607.348,244.014,0,0,1]).reshape(3,3)
        # Determine depth scale
        self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print("camera depth scale:",self.scale)
        print("camera intrinsics:\n",self.intrinsics)
        
        print("D435 connected ...")
    
    def get_intrinsics(self) -> np.array:
        return self.intrinsics
    
    def get_depth_scale(self) -> float:
        return self.scale

    def get_data(self, return_rgb = True):
        # Return color image and depth image
        frames = self.pipeline.wait_for_frames()
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # depth = frames.get_depth_frame()
        # color = frames.get_color_frame()
        # depth_img=np.asarray(depth.get_data())
        # color_img=np.asarray(color.get_data())
        depth_img = np.asanyarray(aligned_depth_frame.get_data())
        bgr_img = np.asanyarray(color_frame.get_data())

        if return_rgb:
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            return rgb_img, depth_img
        else:
            return bgr_img, depth_img
    
    def plot_image(self):
        color_image,depth_image = self.get_data(return_rgb = False)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # depth_colormap_dim = depth_colormap.shape
        # color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
        #                                     interpolation=cv2.INTER_AREA)
        #     images = np.hstack((resized_color_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))
        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', images)

        cv2.namedWindow('RealSense_clolor', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense_clolor', color_image)
        
        cv2.namedWindow('RealSense_depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense_depth', depth_colormap)
        cv2.imwrite('/home/ssm/QCIT/graspness/color_image_5.png', color_image)
        cv2.imwrite('/home/ssm/QCIT/graspness/depth_image_5.png', depth_image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            return True
        else:
            return False

    def plot_image_stream(self):
        while(True):
            if self.plot_image():
                break
            time.sleep(0.1)

if __name__== '__main__':
    mycamera = RealsenseD435()
    mycamera.init_cam()
    # print(mycamera.get_data())
    # mycamera.plot_image()
    while(True):
        mycamera.plot_image()
        time.sleep(0.1)
    # print(mycamera.intrinsics)
    
    
    
    
# VPG
# class Camera(object):
#
#     def __init__(self):
#
#         # Data options (change me)
#         self.im_height = 720
#         self.im_width = 1280
#         self.tcp_host_ip = '127.0.0.1'
#         self.tcp_port = 50000
#         self.buffer_size = 4098 # 4 KiB
#
#         # Connect to server
#         self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
#
#         self.intrinsics = None
#         self.get_data()
#         # color_img, depth_img = self.get_data()
#         # print(color_img, depth_img)
#
#
#     def get_data(self):
#
#         # Ping the server with anything
#         self.tcp_socket.send(b'asdf')
#
#         # Fetch TCP data:
#         #     color camera intrinsics, 9 floats, number of bytes: 9 x 4
#         #     depth scale for converting depth from uint16 to float, 1 float, number of bytes: 4
#         #     depth image, self.im_width x self.im_height uint16, number of bytes: self.im_width x self.im_height x 2
#         #     color image, self.im_width x self.im_height x 3 uint8, number of bytes: self.im_width x self.im_height x 3
#         data = b''
#         while len(data) < (10*4 + self.im_height*self.im_width*5):
#             data += self.tcp_socket.recv(self.buffer_size)
#
#         # Reorganize TCP data into color and depth frame
#         self.intrinsics = np.fromstring(data[0:(9*4)], np.float32).reshape(3, 3)
#         depth_scale = np.fromstring(data[(9*4):(10*4)], np.float32)[0]
#         depth_img = np.fromstring(data[(10*4):((10*4)+self.im_width*self.im_height*2)], np.uint16).reshape(self.im_height, self.im_width)
#         color_img = np.fromstring(data[((10*4)+self.im_width*self.im_height*2):], np.uint8).reshape(self.im_height, self.im_width, 3)
#         depth_img = depth_img.astype(float) * depth_scale
#         return color_img, depth_img


# DEPRECATED CAMERA CLASS FOR REALSENSE WITH ROS
# ----------------------------------------------

# import rospy
# from realsense_camera.msg import StreamData

# class Camera(object):


#     def __init__(self):

#         # Data options (change me)
#         self.im_height = 720
#         self.im_width = 1280

#         # RGB-D data variables
#         self.color_data = np.zeros((self.im_height,self.im_width,3))
#         self.depth_data = np.zeros((self.im_height,self.im_width))
#         self.intrinsics = np.zeros((3,3))

#         # Start ROS subscriber to fetch RealSense RGB-D data
#         rospy.init_node('listener', anonymous=True)
#         rospy.Subscriber("/realsense_camera/stream", StreamData, self.realsense_stream_callback)

#         # Recording variables
#         self.frame_idx = 0
#         self.is_recording = False
#         self.recording_directory = ''

#     # ROS subscriber callback function
#     def realsense_stream_callback(self, data):
#         tmp_color_data = np.asarray(bytearray(data.color))
#         tmp_color_data.shape = (self.im_height,self.im_width,3)
#         tmp_depth_data = np.asarray(data.depth)
#         tmp_depth_data.shape = (self.im_height,self.im_width)
#         tmp_depth_data = tmp_depth_data.astype(float)/10000
#         tmp_intrinsics = np.asarray(data.intrinsics)
#         tmp_intrinsics.shape = (3,3)

#         self.color_data = tmp_color_data
#         self.depth_data = tmp_depth_data
#         self.intrinsics = tmp_intrinsics

#         if self.is_recording:
#             tmp_color_image = cv2.cvtColor(tmp_color_data, cv2.COLOR_RGB2BGR)
#             cv2.imwrite(os.path.join(self.recording_directory, '%06d.color.png' % (self.frame_idx)), tmp_color_image)
#             tmp_depth_image = np.round(tmp_depth_data * 10000).astype(np.uint16) # Save depth in 1e-4 meters
#             cv2.imwrite(os.path.join(self.recording_directory, '%06d.depth.png' % (self.frame_idx)), tmp_depth_image)
#             self.frame_idx += 1
#         else:
#             self.frame_idx = 0

#         time.sleep(0.1)

#     # Start/stop recording RGB-D video stream
#     def start_recording(self, directory):
#         self.recording_directory = directory
#         self.is_recording = True
#     def stop_recording(self):
#         self.is_recording = False

