import warnings
from typing import Optional, Any

import torch
import os

from .. import dataset

# TensorRT 관련 패키지들을 조건부로 임포트
try:
    import tensorrt as trt
    import torch2trt
    from torch2trt import TRTModule
    TENSORRT_AVAILABLE = True
except ImportError as e:
    TENSORRT_AVAILABLE = False
    warnings.warn(f"TensorRT packages not available: {e}. TensorRT functionality will be disabled.")

def check_tensorrt_availability():
    """TensorRT 사용 가능 여부를 확인하는 함수"""
    if not TENSORRT_AVAILABLE:
        raise RuntimeError(
            "TensorRT is not available. Please install tensorrt and torch2trt packages: docs/package.md\n"
        )
    return True

class Int8Calibrator(torch.utils.Data.Dataset):
    """TensorRT INT8 Calibration을 위한 데이터셋 클래스"""
    def __init__(self, data: Any):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]

def convert_to_tensorrt(model, tensorrt_dir):
    """PyTorch 모델을 TensorRT로 변환"""
    try:
        check_tensorrt_availability()
        print("Converting model to TensorRT...")
        backbone = TRTModule()
        root_v2v_net = TRTModule()
        pose_v2v_net = TRTModule()

        backbone.load_state_dict(torch.load(os.path.join(tensorrt_dir, 'backbone.pth'), weights_only=False))
        root_v2v_net.load_state_dict(torch.load(os.path.join(tensorrt_dir, 'root_v2v_net.pth'), weights_only=False))
        pose_v2v_net.load_state_dict(torch.load(os.path.join(tensorrt_dir, 'pose_v2v_net.pth'), weights_only=False))

        model.backbone = backbone
        model.root_v2v_net = root_v2v_net
        model.pose_v2v_net = pose_v2v_net
        print("Model converted to TensorRT successfully.")
        return model

    except Exception as e:
        warnings.warn(f"Failed to convert model to TensorRT, using original model: {e}")
        return model