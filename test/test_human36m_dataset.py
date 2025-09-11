#!/usr/bin/env python3
"""
Human36M 데이터셋을 SelfPose3d로 학습하는 테스트 스크립트
"""

import os
import sys
import argparse

# SelfPose3d 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SelfPose3d.core.config import config, update_config
from SelfPose3d.dataset import human36m_ssv, human36m


def parse_args():
    parser = argparse.ArgumentParser(description='Test Human36M SSV Dataset')
    parser.add_argument('--cfg', 
                       help='experiment configure file name',
                       default='configs/human36m/backbone_h36m.yaml',
                       type=str)
    parser.add_argument('opts',
                       help="Modify config options using the command-line",
                       default=None,
                       nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def test_dataset():
    """데이터셋 로딩 테스트"""
    args = parse_args()
    update_config(args.cfg)
    
    print("==> Testing Human36M SSV Dataset")
    print(f"Config file: {args.cfg}")
    
    try:
        # 데이터셋 생성
        train_dataset = human36m(
            config, 
            image_set='train',
            is_train=True,
            transform=None
        )
        
        print(f"✅ Train dataset loaded successfully!")
        print(f"   - Total samples: {len(train_dataset)}")
        print(f"   - Number of views: {train_dataset.num_views}")
        print(f"   - Number of joints: {train_dataset.num_joints}")
        print(f"   - Cameras: {train_dataset.cameras}")
        
        # 첫 번째 샘플 로드 테스트
        if len(train_dataset) > 0:
            print("\n==> Testing first sample loading...")
            try:
                sample = train_dataset[0]
                print(f"✅ First sample loaded successfully!")
                print(f"   - Sample type: {type(sample)}")
                if isinstance(sample, tuple):
                    print(f"   - Sample length: {len(sample)}")
            except Exception as e:
                print(f"❌ Error loading first sample: {e}")
    
        # 검증 데이터셋도 테스트
        val_dataset = human36m(
            config, 
            image_set='validation',
            is_train=False,
            transform=None
        )
        
        print(f"\n✅ Validation dataset loaded successfully!")
        print(f"   - Total samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = test_dataset()
    if success:
        print("\n🎉 Human36M SSV dataset is ready for SelfPose3d training!")
        print("\nNext steps:")
        print("1. Ensure Human36M data is properly organized in data/Human36M/")
        print("2. Run training with: python tools/train_3d.py --cfg configs/human36m/resnet50_human36m_4cam.yaml")
    else:
        print("\n💥 Dataset test failed. Please check the configuration and data paths.")
        sys.exit(1)
