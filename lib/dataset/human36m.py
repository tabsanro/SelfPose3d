'''
Project: SelfPose3d - Human36M Multi-view Dataset
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import os.path as osp
import numpy as np
import math
import torch
import json
import copy
import logging
import pickle

from .JointsDataset import JointsDataset
from ..utils.cameras_cpu import camera_to_world_frame, project_pose

logger = logging.getLogger(__name__)

# Human36M 조인트 정의 (17개)
JOINTS_DEF = {
    "pelvis": 0,
    "r_hip": 1, 
    "r_knee": 2,
    "r_ankle": 3,
    "l_hip": 4,
    "l_knee": 5,
    "l_ankle": 6,
    "torso": 7,
    "neck": 8,
    "nose": 9,
    "head": 10,
    "l_shoulder": 11,
    "l_elbow": 12,
    "l_wrist": 13,
    "r_shoulder": 14,
    "r_elbow": 15,
    "r_wrist": 16,
}

# Human36M flip pairs (17개 조인트)
FLIP_LR_JOINTS17 = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]

LIMBS = [
    [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], 
    [8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], 
    [0, 4], [4, 5], [5, 6]
]

# Human36M 카메라 설정 (4개 카메라)
TRAIN_CAMERAS = [0, 1, 2, 3]  # 모든 카메라 사용
VAL_CAMERAS = [0, 1, 2, 3]    # 모든 카메라 사용


class Human36M(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)
        
        # Human36M specific settings
        dataset_name = 'Human36M'
        self.data_split = image_set
        self.img_dir = osp.join(cfg.DATASET.ROOT, dataset_name, 'images')
        self.annot_path = osp.join(cfg.DATASET.ROOT, dataset_name, 'annotations')
        
        self.subject_genders = {1: 'female', 5: 'female', 6: 'male', 7: 'female', 8: 'male', 9: 'male', 11: 'male'}
        self.protocol = 2
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                            'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog',
                            'WalkTogether']
        self.fitting_thr = 25  # milimeter

        # H36M joint set
        self.human36_joint_num = 17
        self.human36_joints_name = (
        'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.human36_skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
        self.human36_root_joint_idx = self.human36_joints_name.index('Pelvis')
        self.human36_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)

        # 카메라 설정
        if self.image_set == "train":
            self.cameras = TRAIN_CAMERAS
        else:
            self.cameras = VAL_CAMERAS
            
        self.num_views = len(self.cameras)
        self.camera_num_total = len(self.cameras)
        
        # SelfPose3d 데이터베이스 파일 생성
        self.db_file = "human36m_{}_cam{}_ssv.pkl".format(
            self.image_set, self.camera_num_total
        )
        self.db_file = osp.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file):
            info = torch.load(self.db_file)
            self.db = info['db']
            logger.info("=> load db length: {}".format(len(self.db)))
        else:
            self.db = self._get_db()
            info = {
                'db': self.db
            }
            torch.save(info, self.db_file)
            logger.info("=> dump db file: {}".format(self.db_file))
            logger.info("=> dump db length: {}".format(len(self.db)))

        self.db_size = len(self.db)

    def get_subject(self):
        if self.data_split == 'train':
            if self.protocol == 1:
                subject = [1, 5, 6, 7, 8, 9]
            elif self.protocol == 2:
                subject = [1, 5, 6, 7, 8]
        elif self.data_split == 'test' or self.data_split == 'validation':
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9, 11]
        else:
            assert 0, print("Unknown subset")
        return subject

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test' or self.data_split == 'validation':
            return 50
        else:
            assert 0, print('Unknown subset')

    def _get_cam(self, camera_param):
        """카메라 파라미터를 SelfPose3d 형식으로 변환"""
        R, t, f, c = camera_param['R'], camera_param['t'], camera_param['f'], camera_param['c']
        R = np.array(R, dtype=np.float32)
        t = np.array(t, dtype=np.float32)
        f = np.array(f, dtype=np.float32)
        c = np.array(c, dtype=np.float32)
        
        our_cam = {
            "R": R,
            "T": -np.dot(R.T, t).reshape(3, 1),  # mm 단위로 변환
            "fx": np.array(f[0], dtype=np.float32),  # 스칼라를 배열로 변환
            "fy": np.array(f[1], dtype=np.float32),  # 스칼라를 배열로 변환
            "cx": np.array(c[0], dtype=np.float32),  # 스칼라를 배열로 변환
            "cy": np.array(c[1], dtype=np.float32),  # 스칼라를 배열로 변환
            "k": np.zeros((3, 1), dtype=np.float32),  # (3,1) shape
            "p": np.zeros((2, 1), dtype=np.float32)   # (2,1) shape
        }
        return our_cam

    def _get_db(self):
        """Human36M 데이터를 SelfPose3d 형식으로 변환"""
        print('Load annotations of Human36M Protocol ' + str(self.protocol))
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        db = []
        
        for subject in subject_list:
            print(f"Processing subject {subject}...")
            
            # 데이터 로드
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'), 'r') as f:
                data_annot = json.load(f)
            
            # 카메라 로드
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'), 'r') as f:
                cameras = json.load(f)
            
            # 관절 좌표 로드
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
                joints = json.load(f)
            
            # 이미지와 어노테이션 매핑
            images = data_annot.get('images', [])
            annotations = data_annot.get('annotations', [])
            
            # 이미지 ID를 키로 하는 딕셔너리 생성
            img_dict = {img['id']: img for img in images}
            
            for ann in annotations:
                image_id = ann['image_id']
                if image_id not in img_dict:
                    continue
                    
                img = img_dict[image_id]
                
                # 프레임 샘플링 체크
                frame_idx = img['frame_idx']
                if frame_idx % sampling_ratio != 0:
                    continue
                    
                subject_id = img['subject']
                action_idx = img['action_idx']
                subaction_idx = img['subaction_idx']
                cam_idx = img['cam_idx']
                
                # 카메라 필터링 - 선택된 카메라만 사용
                if cam_idx not in self.cameras:
                    continue

                # 카메라 파라미터 변환
                cam_param = cameras[str(cam_idx)]
                our_cam = self._get_cam(cam_param)
                
                # 관절 데이터 확인
                try:
                    joints_3d = np.array(joints[str(action_idx)][str(subaction_idx)][str(frame_idx)])
                    joints_3d = \
                        camera_to_world_frame(
                            joints_3d,
                            our_cam['R'],
                            our_cam['T']
                        )
                    joints_2d = project_pose(joints_3d, our_cam)
                except KeyError:
                    continue
                
                # 이미지 경로
                img_path = osp.join(self.img_dir, img['file_name'])
                
                # 3D 조인트 (Human36M에서는 이미 카메라 좌표계)
                joints_3d = np.array(joints_3d, dtype=np.float32).reshape(-1, 3)
                
                # 단일 사람만 처리 (Human36M은 단일 사람 데이터셋)
                if len(joints_3d) != self.num_joints:
                    continue
                
                # bbox 처리
                bbox = np.array(ann['bbox'])
                if len(bbox) == 4:  # [x, y, w, h] 형식
                    bbox = self._process_bbox(bbox)
                    if bbox is None:
                        continue
                
                # visibility 설정 (모든 조인트가 보인다고 가정)
                joints_vis = np.ones((self.num_joints,))
                
                # panoptic_ssv와 동일한 형식으로 데이터 구성
                db.append({
                    "key": f"s{subject_id:02d}_act{action_idx:02d}_subact{subaction_idx:02d}_cam{cam_idx:02d}_frame{frame_idx:06d}",
                    "image": img_path,
                    "joints_3d": [joints_3d],  # 리스트로 감싸기 (다중 사람 형식)
                    "joints_3d_vis": [np.ones((self.num_joints, 3))],  # 3D visibility
                    "joints_2d": [joints_2d],  # 리스트로 감싸기
                    "joints_2d_vis": [np.ones((self.num_joints, 2))],  # 2D visibility
                    "camera": our_cam,
                })
        
        print(f"Total samples: {len(db)}")
        return db

    def _project_3d_to_2d(self, joints_3d, camera):
        """3D 조인트를 2D로 프로젝션"""
        fx, fy = camera['fx'], camera['fy']
        cx, cy = camera['cx'], camera['cy']
        
        joints_2d = np.zeros((len(joints_3d), 2))
        for i, joint in enumerate(joints_3d):
            if joint[2] > 0:  # z > 0인 경우만
                joints_2d[i, 0] = joint[0] * fx / joint[2] + cx
                joints_2d[i, 1] = joint[1] * fy / joint[2] + cy
        
        return joints_2d

    def _process_bbox(self, bbox):
        """bbox 처리 (x, y, w, h -> 유효한 bbox)"""
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        return np.array([x, y, w, h])

    def _get_scale(self, bbox):
        """bbox에서 스케일 계산"""
        return max(bbox[2], bbox[3]) / 200.0

    def _get_center(self, bbox):
        """bbox에서 중심점 계산"""
        return np.array([bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5])

    def __len__(self):
        return self.db_size // self.num_views

    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for k in range(self.num_views):
            i, t, w, t3, m, ih = super().__getitem__(self.camera_num_total * idx + self.cameras[k])
            if i is None:
                continue
            input.append(i)
            target.append(t)
            weight.append(w)
            target_3d.append(t3)
            meta.append(m)
            input_heatmap.append(ih)
        return input, target, weight, target_3d, meta, input_heatmap

    def evaluate(self, preds):
        """SelfPose3d 스타일 평가 (panoptic_ssv와 동일한 구조)"""
        eval_list = []
        gt_num = self.db_size // self.num_views
        assert len(preds) == gt_num, "number mismatch"

        total_gt = 0
        for i in range(gt_num):
            index = self.num_views * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec["joints_3d"]
            joints_3d_vis = db_rec["joints_3d_vis"]

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]  # valid한 예측만 선택
            
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(
                        np.sqrt(
                            np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)
                        )
                    )
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4] if pose.shape[1] > 4 else 1.0  # Human36M에는 score가 없을 수 있음
                eval_list.append(
                    {
                        "mpjpe": float(min_mpjpe),
                        "score": float(score),
                        "gt_id": int(total_gt + min_gt),
                    }
                )

            total_gt += len(joints_3d)

        # 임계값별 평가
        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return (
            aps,
            recs,
            self._eval_list_to_mpjpe(eval_list),
            self._eval_list_to_recall(eval_list, total_gt),
        )

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        """정확도(AP) 계산 (panoptic_ssv와 동일)"""
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        """MPJPE 계산 (panoptic_ssv와 동일)"""
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        """Recall 계산 (panoptic_ssv와 동일)"""
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt