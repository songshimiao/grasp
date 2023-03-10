import os
import sys
import numpy as np
import argparse
from PIL import Image
# import time
# import scipy.io as scio
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# print(ROOT_DIR)
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
#default='realsense_1120_epoch10.tar')

parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
cfgs = parser.parse_args()

class Graspness():
    def __init__(self):
        self.rgbd_cam_d435 = RealsenseD435()
        self.rgbd_cam_d435.init_cam()
        self.net = None
        self.device = None
        self.init_grasp_net()
        print("init success!")
    
    def get_color_depth_data(self):
        self.rgbd_cam_d435.plot_image_stream() #按q退出视频流
        return self.rgbd_cam_d435.get_data()
    
    def data_process(self):
        # load real data
        intrinsic = self.rgbd_cam_d435.get_intrinsics()
        factor_depth = 1.0 / self.rgbd_cam_d435.get_depth_scale()
        
        root='/home/ssm/QCIT/graspness/'
        rgb, depth = self.get_color_depth_data()
        color = rgb.astype(np.float32) / 255
        depth = depth.astype(np.float32)

        # workspace_mask = np.array(Image.open(os.path.join(root,'doc/example_data', 'mask_all.png'))).astype(bool) #[720,1280][241false,978false]
        workspace_mask = np.array(Image.open(os.path.join(root,'doc/example_data', 'workspace_mask.png'))) #[720,1280][241false,978false]
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        # compute workspace limits
        # x_left_up,y_left_up = 0+25,242+25
        # x_right_bottom,y_right_bottom = 480-25,640-25
        # x_left_up,y_left_up = 0+25,242+25
        # x_right_bottom,y_right_bottom = 719-25,977-25
        x_left_up,y_left_up = 0+50,0+50
        x_right_bottom,y_right_bottom = 720-50,1280-50
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

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)#720,1080,3
        # print("cloud 维度： {}".format(cloud.shape))

        # get valid points
        depth_mask = (depth > 0) #depth_mask存放深度有效的索引
        # camera_poses = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'camera_poses.npy'))
        # align_mat = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
        # trans = np.dot(align_mat, camera_poses[int(index)])
        # workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (workspace_mask & depth_mask)

        cloud_masked = cloud[mask]#51225,3
        color_masked = color[mask]
        # sample points random
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
        print("cloud o3d 大小： {}".format(len(cloud.points)))
        # o3d.visualization.draw_geometries([cloud])


        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': (cloud_sampled.astype(np.float32) / cfgs.voxel_size).astype(np.float32),
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    }
        return ret_dict,cloud,point_left_up,point_right_bottom

    def init_grasp_net(self):
        self.net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("设备：{}".format(self.device))
        self.net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(cfgs.checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
        self.net.eval()   

    def grasp(self, data_input,cloud_,point_left_up,point_right_bottom):
        batch_data = minkowski_collate_fn([data_input])
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self.device)
            else:
                batch_data[key] = batch_data[key].to(self.device)
        # Forward pass
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds = pred_decode(end_points)#1024,17
        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)

        # collision detection
        if cfgs.collision_thresh > 0:
            cloud = data_input['point_clouds']
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            gg = gg[~collision_mask]
        if len(gg)==0:
            print("detect nothing or have no grasp pose.")
            return False
        gg.nms()
        gg.sort_by_score()
        if len(gg)>50:
            gg = gg[:50]

        # 剔除不在workspace下的grasp
        for i in range(len(gg)-1,-1,-1):
            if gg[i].translation[0]< point_left_up[0]+0.02 or gg[i].translation[0] >point_right_bottom[0]-0.02\
                    or gg[i].translation[1]<point_left_up[1]+0.02 or gg[i].translation[1]>point_right_bottom[1]-0.02:
                gg.remove(i)

        if len(gg)==0:
            print("detect nothing or have no grasp pose ")
            return False
        gg.sort_by_score()
        
        grippers = gg.to_open3d_geometry_list()
        grippers[0].paint_uniform_color([0, 1, 0])  # the best score grasp pose's color  is green
        # o3d.visualization.draw_geometries([cloud_, *grippers],
        #                                   lookat=[1.0, 2.0, 2.0],
        #                                   front=[50.0, 50.0, 50.0],
        #                                   up=[50.0, 50.0, 50.0],
        #                                   zoom=200.0)
        o3d.visualization.draw_geometries([cloud_, *grippers])


        R_grasp2camera, t_grasp2camera = gg[0].rotation_matrix, gg[0].translation
        print(R_grasp2camera,t_grasp2camera)
        return True

if __name__ == '__main__':
    
    grasp_demo = Graspness()
    while True: 
        a,b,c,d = grasp_demo.data_process()
        grasp_demo.grasp(a,b,c,d)
    
    
    
    
    
    
    
    
    
    

    # ur_robot = UR_Robot(tcp_host_ip="192.168.50.100", tcp_port=30003, workspace_limits=None, is_use_robotiq85=True,
    #                     is_use_camera=True)
    # grasp_result = []
    # iter = 0
    # while True:
    # #for i in range(50):
    #     data_dict, cloud, point_left_up, point_right_bottom = data_process()
    #     grasp_success = grasp(data_dict, cloud, point_left_up, point_right_bottom)
    #     print(grasp_success)
    #     if grasp_success:
    #         grasp_result.append(True)
    #     else:
    #         grasp_result.append(False)
    #     # end
    #     if (iter >= 2) and (not grasp_result[iter]) and (not grasp_result[iter - 1]) and (not grasp_result[iter - 2]):
    #         print('grasp_result_array:', grasp_result)
    #         print("finish...")
    #         break
    #     iter += 1
    
    
