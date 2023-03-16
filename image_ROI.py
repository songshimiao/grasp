################################
#   image_ROI
#   输入为图片数据
#   手动框选想要提取抓握姿态的目标
################################

import os
import sys
import cv2
import time
import numpy as np
import argparse
from PIL import Image
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('[INFO] ROOT_DIR = {}'.format(ROOT_DIR))

# 将所需路径导入系统路径
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
# from UR_Robot import UR_Robot
from realsenseD435 import RealsenseD435

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=ROOT_DIR+'/np15000_graspness1e-1_bs4_lr1e-3_viewres_dataaug_fps_14D_epoch10.tar')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
cfgs = parser.parse_args()
drawing = False
ROI_left_up_point = None
ROI_right_bottom_point = None
img = None

class ImageInfo():
    def __init__(self):
        self.im_height = 720
        self.im_width = 1280
    
    def get_data(self):
        # color_image = cv2.imread(ROOT_DIR+'/color_image_self.png')
        # depth_image = cv2.imread(ROOT_DIR+'/depth_image_self.png')
        # depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        color_image = np.array(Image.open(os.path.join(ROOT_DIR, 'color_image_self.png')))
        depth_image = np.array(Image.open(os.path.join(ROOT_DIR, 'depth_image_self.png')))

        return color_image, depth_image
    
    def plot_image(self):
        color_image, depth_image = self.get_data()
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('depth image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('color image', color_image)
        cv2.imshow('depth image', depth_image)
        key = cv2.waitKey()
        print('[INFO] Loaded images data success.')
        
        
        
class CameraInfo():
    '''Camera intrisics for point cloud creation.'''
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    assert (depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    print('[INFO] xmap.shape={}, ymap.shape={}'.format(xmap.shape, ymap.shape))
    point_z = depth / camera.scale
    point_x = (xmap - camera.cx) * point_z / camera.fx
    point_y = (ymap - camera.cy) * point_z / camera.fy
    cloud = np.stack([point_x, point_y, point_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud



class Graspness():
    def __init__(self):
        self.net = None
        self.device = None
        self.image = ImageInfo()
        self.intrinsics = np.array([905.70556641, 0., 640.300354, 0., 903.98883057, 357.47369385, 0., 0., 1.]).reshape(3,3)
        self.scale = 0.001
        print('[INFO] camera depth scale:', self.scale)
        print('[INFO] camera intrinsics:\n', self.intrinsics)
        self.init_grasp_net()
        print('[INFO] Init success.')
        
    def creat_ROI(self, color_image) -> tuple:
        '''Input: rgb_image.    Output: ROI_left_up_point, ROI_right_bottom_point'''
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        global img
        img = color_image.copy()
        def draw_rectangle(event, x, y, flags, param):
            global ROI_left_up_point, ROI_right_bottom_point
            global drawing
            global img
            img = color_image.copy()
            cv2.imshow('color_image', img)
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ROI_left_up_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    cv2.rectangle(img, ROI_left_up_point, (x, y), (0,0,255), thickness=2)
                    cv2.imshow('color_image', img)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                ROI_right_bottom_point = (x, y)
                cv2.rectangle(color_image, ROI_left_up_point, (x, y), (0,0,255), thickness=2)
                cv2.imshow('color_image', img)
        
        cv2.namedWindow('color_image', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('color_image', draw_rectangle)
        while(True):
            if not drawing:
                cv2.imshow('color_image', color_image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
        cv2.destroyAllWindows() 
        print('ROI区域:', ROI_left_up_point, ROI_right_bottom_point)
        return ROI_left_up_point, ROI_right_bottom_point
    
    def get_color_depth_data(self):
        # self.image.plot_image() #按任意键跳出   waitkey()
        return self.image.get_data()
    
    
    def data_process(self):
        intrinsic = self.intrinsics
        factor_depth = 1.0 / self.scale
        
        root = '/home/ssm/QCIT/graspness/'
        rgb, depth = self.get_color_depth_data()
        color = rgb.astype(np.float32) / 255
        depth = depth.astype(np.float32)

        ROI_left_up_point, ROI_right_bottom_point = self.creat_ROI(rgb)

        print('[INFO] color shape:{}, depth shape:{}'.format(color.shape, depth.shape))
        
        workspace_mask = np.array(Image.open(os.path.join(root, 'doc/example_data', 'workspace_mask.png')))
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        
        # x_left_up,y_left_up = 0+50,0+50
        # x_right_bottom,y_right_bottom = 720-50,1280-50
        x_left_up,y_left_up = ROI_left_up_point[1],ROI_left_up_point[0]
        x_right_bottom,y_right_bottom = ROI_right_bottom_point[1],ROI_right_bottom_point[0]        
        point_z = depth[x_left_up,y_left_up] / camera.scale
        point_x = (y_left_up - camera.cx) * point_z / camera.fx
        point_y = (x_left_up - camera.cy) * point_z / camera.fy
        point_left_up = (point_x,point_y,point_z)
        point_z = depth[x_right_bottom,y_right_bottom] / camera.scale
        point_x = (y_right_bottom - camera.cx) * point_z / camera.fx
        point_y = (x_right_bottom - camera.cy) * point_z / camera.fy
        point_right_bottom = (point_x, point_y, point_z)
        
        for x in range(x_left_up, x_right_bottom):
            for y in range(y_left_up-2, y_left_up+3):
                color[x][y] = [1,0,0]
                
        for x in range(x_left_up, x_right_bottom):
            for y in range(y_right_bottom-2, y_right_bottom+3):
                color[x][y] = [0,1,0]
                
        for x in range(x_left_up-2, x_left_up+2):
            for y in range(y_left_up, y_right_bottom):
                color[x][y] = [0,0,1]
                
        for x in range(x_right_bottom-2, x_right_bottom+2):
            for y in range(y_left_up, y_right_bottom):
                color[x][y] = [0,0,0]
                
                
        # 生成点云
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)#720,1080,3
        print('[INFO] cloud 维度：{}'.format(cloud.shape))
        
        depth_mask = (depth > 0)
        mask = (workspace_mask & depth_mask)
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        
        if len(cloud_masked) >= cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]#15000,3
        color_sampled = color_masked[idxs]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32)) #51w points
        print("[INFO] cloud o3d 大小： {}".format(len(cloud.points)))
        
        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': (cloud_sampled.astype(np.float32) / cfgs.voxel_size).astype(np.float32),
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    }
        return ret_dict,cloud,point_left_up,point_right_bottom
        
        
        
    def init_grasp_net(self):
        self.net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('[INFO] 设备：{}'.format(self.device))
        self.net.to(self.device)
        checkpoint = torch.load(cfgs.checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print('[INFO] loaded checkpoint %s (epoch: %d)' % (cfgs.checkpoint_path, start_epoch))
        self.net.eval()

    def grasp(self, data_input, cloud_, point_left_up, point_right_bottom):
        batch_data = minkowski_collate_fn([data_input])
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self.device)
            else:
                batch_data[key] = batch_data[key].to(self.device)
                
        # 前向传播
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds = pred_decode(end_points)
        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)
        
        # collision detection
        if cfgs.collision_thresh > 0:
            cloud = data_input['point_clouds']
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            gg = gg[~collision_mask]
        if len(gg) == 0:
            print('[INFO] detect nothing or have no grasp pose.')
            return False
        gg.nms()
        gg.sort_by_score()
        if len(gg) > 50:
            gg = gg[:50]
            
        # 剔除不在工作区的抓握姿态
        for i in range(len(gg)-1, -1, -1):
            if gg[i].translation[0]< point_left_up[0]+0.02 or gg[i].translation[0] > point_right_bottom[0]-0.02\
                    or gg[i].translation[1]<point_left_up[1]+0.02 or gg[i].translation[1] > point_right_bottom[1]-0.02:
                gg.remove(i)
                
        if len(gg) == 0:
            print('[INFO] detect nothing or have no grasp pose')
            return False
        gg.sort_by_score()
        print('[INFO]workspace:{}'.format(len(gg)))
        
        grippers = gg.to_open3d_geometry_list()
        grippers[0].paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([cloud_, *grippers])
        
        
        R_grasp2camera, t_grasp2camera = gg[0].rotation_matrix, gg[0].translation
        width = gg[0].width * 10 + 0.05
        print(R_grasp2camera,t_grasp2camera,width)
        
        return True

        
        
        
        
        
if __name__ == '__main__':
    image_demo = Graspness()
    while(True):
        a, b, c, d = image_demo.data_process()
        image_demo.grasp(a,b,c,d)

