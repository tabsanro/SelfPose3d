import cv2
import numpy as np
import os
import glob
import argparse
from typing import List, Tuple, Optional


class CameraCalibrator:
    """
    카메라 캘리브레이션 클래스
    
    Step1: 체커보드를 이용한 내부 파라미터 캘리브레이션
    Step2: 아루코 마커를 이용한 외부 파라미터 캘리브레이션
    
    권장 체커보드 패턴 다운로드:
    - OpenCV 공식: https://docs.opencv.org/4.x/pattern.png
    - 또는 직접 생성: 9x6 체커보드 (8x5 내부 코너)
    """
    
    def __init__(self, checkerboard_size: Tuple[int, int] = (8, 5), square_size: float = 0.025):
        """
        Args:
            checkerboard_size: 체커보드 내부 코너 개수 (가로, 세로)
            square_size: 체커보드 한 칸의 실제 크기 (미터 단위, 기본값: 2.5cm)
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # 체커보드 3D 좌표 생성
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # 캘리브레이션 데이터 저장
        self.objpoints = []  # 3D 월드 좌표
        self.imgpoints = []  # 2D 이미지 좌표
        
        # 캘리브레이션 결과
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
    def detect_checkerboard_corners(self, image: np.ndarray, visualize: bool = False) -> Tuple[bool, Optional[np.ndarray]]:
        """
        체커보드 코너 검출
        
        Args:
            image: 입력 이미지
            visualize: 검출 결과 시각화 여부
            
        Returns:
            success: 검출 성공 여부
            corners: 검출된 코너 좌표
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 체커보드 코너 검출
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            # 서브픽셀 정확도로 코너 정제
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            if visualize:
                img_with_corners = image.copy()
                cv2.drawChessboardCorners(img_with_corners, self.checkerboard_size, corners, ret)
                cv2.imshow('Checkerboard Corners', img_with_corners)
                cv2.waitKey(500)
        
        return ret, corners if ret else None
    
    def collect_calibration_data(self, image_paths: List[str], visualize: bool = False) -> int:
        """
        캘리브레이션 데이터 수집
        
        Args:
            image_paths: 체커보드 이미지 경로 리스트
            visualize: 검출 과정 시각화 여부
            
        Returns:
            성공적으로 처리된 이미지 개수
        """
        self.objpoints.clear()
        self.imgpoints.clear()
        
        successful_images = 0
        
        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Cannot read image {img_path}")
                continue
                
            ret, corners = self.detect_checkerboard_corners(image, visualize)
            
            if ret:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
                successful_images += 1
                print(f"✓ Processed: {os.path.basename(img_path)}")
            else:
                print(f"✗ Failed to detect corners: {os.path.basename(img_path)}")
        
        if visualize:
            cv2.destroyAllWindows()
            
        return successful_images
    
    def calibrate_intrinsic(self, image_shape: Tuple[int, int]) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        내부 파라미터 캘리브레이션 수행
        
        Args:
            image_shape: 이미지 크기 (높이, 너비)
            
        Returns:
            reprojection_error: 재투영 오차
            camera_matrix: 카메라 내부 파라미터 행렬
            dist_coeffs: 왜곡 계수
        """
        if len(self.objpoints) == 0:
            raise ValueError("No calibration data available. Run collect_calibration_data() first.")
        
        # 캘리브레이션 수행
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, 
            (image_shape[1], image_shape[0]),  # (width, height)
            None, None
        )
        
        # 결과 저장
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        return ret, mtx, dist
    
    def save_intrinsic_parameters(self, filepath: str):
        """내부 파라미터를 파일로 저장"""
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("No calibration results available.")
        
        # NumPy 형식으로 저장
        np.savez(filepath, 
                 camera_matrix=self.camera_matrix,
                 dist_coeffs=self.dist_coeffs,
                 image_shape=self.imgpoints[0].shape if self.imgpoints else None)
        
        print(f"Intrinsic parameters saved to: {filepath}")
        
    def load_intrinsic_parameters(self, filepath: str):
        """저장된 내부 파라미터 로드"""
        data = np.load(filepath)
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']
        print(f"Intrinsic parameters loaded from: {filepath}")
    
    def print_intrinsic_results(self, reprojection_error: float):
        """내부 파라미터 캘리브레이션 결과 출력"""
        print("\n" + "="*60)
        print("카메라 내부 파라미터 캘리브레이션 결과")
        print("="*60)
        print(f"재투영 오차 (Reprojection Error): {reprojection_error:.4f} pixels")
        print(f"사용된 이미지 개수: {len(self.objpoints)}")
        print("\n카메라 내부 파라미터 행렬 (Camera Matrix):")
        print(self.camera_matrix)
        print(f"\n초점거리 (Focal Length): fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
        print(f"주점 (Principal Point): cx={self.camera_matrix[0,2]:.2f}, cy={self.camera_matrix[1,2]:.2f}")
        print("\n왜곡 계수 (Distortion Coefficients):")
        print(f"k1={self.dist_coeffs[0][0]:.6f}, k2={self.dist_coeffs[0][1]:.6f}")
        print(f"p1={self.dist_coeffs[0][2]:.6f}, p2={self.dist_coeffs[0][3]:.6f}")
        if len(self.dist_coeffs[0]) > 4:
            print(f"k3={self.dist_coeffs[0][4]:.6f}")
        print("="*60)
    
    def detect_aruco_markers(self, image: np.ndarray, aruco_dict_type=cv2.aruco.DICT_6X6_250, 
                           visualize: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        아루코 마커 검출
        
        Args:
            image: 입력 이미지
            aruco_dict_type: 아루코 딕셔너리 타입
            visualize: 검출 결과 시각화 여부
            
        Returns:
            corners: 검출된 마커 코너들
            ids: 검출된 마커 ID들
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 아루코 딕셔너리 생성
        aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
        aruco_params = cv2.aruco.DetectorParameters_create()
        
        # 마커 검출
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        if visualize and ids is not None:
            img_with_markers = image.copy()
            cv2.aruco.drawDetectedMarkers(img_with_markers, corners, ids)
            cv2.imshow('ArUco Markers', img_with_markers)
            cv2.waitKey(500)
        
        return corners if ids is not None else None, ids
    
    def estimate_pose_single_marker(self, image: np.ndarray, marker_size: float = 0.05,
                                  aruco_dict_type=cv2.aruco.DICT_6X6_250, 
                                  visualize: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        단일 아루코 마커로부터 포즈 추정
        
        Args:
            image: 입력 이미지
            marker_size: 마커의 실제 크기 (미터 단위)
            aruco_dict_type: 아루코 딕셔너리 타입
            visualize: 결과 시각화 여부
            
        Returns:
            poses: [(rvec, tvec), ...] 형태의 포즈 리스트
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera intrinsic parameters not available. Run calibrate_intrinsic() first.")
        
        corners, ids = self.detect_aruco_markers(image, aruco_dict_type, False)
        poses = []
        
        if corners is not None:
            # 각 마커에 대해 포즈 추정
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_size, self.camera_matrix, self.dist_coeffs
            )
            
            for i in range(len(ids)):
                poses.append((rvecs[i][0], tvecs[i][0]))
                
            if visualize:
                img_with_pose = image.copy()
                for i in range(len(ids)):
                    # 좌표축 그리기
                    cv2.aruco.drawAxis(img_with_pose, self.camera_matrix, self.dist_coeffs,
                                     rvecs[i], tvecs[i], marker_size)
                    # 마커 그리기
                    cv2.aruco.drawDetectedMarkers(img_with_pose, corners, ids)
                    
                    # 포즈 정보 텍스트로 표시
                    tvec = tvecs[i][0]
                    text = f"ID:{ids[i][0]} x:{tvec[0]:.3f} y:{tvec[1]:.3f} z:{tvec[2]:.3f}"
                    cv2.putText(img_with_pose, text, (10, 30 + i * 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('ArUco Pose Estimation', img_with_pose)
                cv2.waitKey(1000)
        
        return poses
    
    def estimate_pose_board(self, image: np.ndarray, board_size: Tuple[int, int] = (5, 7),
                          marker_size: float = 0.04, marker_separation: float = 0.01,
                          aruco_dict_type=cv2.aruco.DICT_6X6_250, 
                          visualize: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        아루코 보드로부터 포즈 추정
        
        Args:
            image: 입력 이미지
            board_size: 보드 크기 (가로, 세로 마커 개수)
            marker_size: 마커의 실제 크기 (미터 단위)
            marker_separation: 마커 간 간격 (미터 단위)
            aruco_dict_type: 아루코 딕셔너리 타입
            visualize: 결과 시각화 여부
            
        Returns:
            pose: (rvec, tvec) 또는 None
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera intrinsic parameters not available. Run calibrate_intrinsic() first.")
        
        # 아루코 보드 생성
        aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
        board = cv2.aruco.GridBoard_create(
            board_size[0], board_size[1], marker_size, marker_separation, aruco_dict
        )
        
        corners, ids = self.detect_aruco_markers(image, aruco_dict_type, False)
        
        if corners is not None and len(corners) > 0:
            # 보드 포즈 추정
            retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                corners, ids, board, self.camera_matrix, self.dist_coeffs, None, None
            )
            
            if retval > 0:  # 성공적으로 포즈 추정된 경우
                if visualize:
                    img_with_pose = image.copy()
                    # 좌표축 그리기 (보드 크기의 절반 길이)
                    axis_length = max(marker_size, marker_separation) * max(board_size) * 0.5
                    cv2.aruco.drawAxis(img_with_pose, self.camera_matrix, self.dist_coeffs,
                                     rvec, tvec, axis_length)
                    # 검출된 마커들 그리기
                    cv2.aruco.drawDetectedMarkers(img_with_pose, corners, ids)
                    
                    # 포즈 정보 텍스트로 표시
                    text = f"Board Pose - x:{tvec[0][0]:.3f} y:{tvec[1][0]:.3f} z:{tvec[2][0]:.3f}"
                    cv2.putText(img_with_pose, text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('ArUco Board Pose', img_with_pose)
                    cv2.waitKey(1000)
                
                return rvec, tvec
        
        return None
    
    def save_extrinsic_parameters(self, rvec: np.ndarray, tvec: np.ndarray, filepath: str):
        """외부 파라미터를 파일로 저장"""
        # 회전 벡터를 회전 행렬로 변환
        R, _ = cv2.Rodrigues(rvec)
        
        # 4x4 변환 행렬 생성
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        np.savez(filepath, 
                 rvec=rvec,
                 tvec=tvec,
                 rotation_matrix=R,
                 transformation_matrix=T)
        
        print(f"Extrinsic parameters saved to: {filepath}")
    
    def print_extrinsic_results(self, rvec: np.ndarray, tvec: np.ndarray):
        """외부 파라미터 결과 출력"""
        # 회전 벡터를 회전 행렬로 변환
        R, _ = cv2.Rodrigues(rvec)
        
        print("\n" + "="*60)
        print("카메라 외부 파라미터 (카메라 포즈)")
        print("="*60)
        print("회전 벡터 (Rotation Vector):")
        print(f"rx={rvec[0][0]:.6f}, ry={rvec[1][0]:.6f}, rz={rvec[2][0]:.6f}")
        print("\n이동 벡터 (Translation Vector) [미터]:")
        print(f"tx={tvec[0][0]:.6f}, ty={tvec[1][0]:.6f}, tz={tvec[2][0]:.6f}")
        print("\n회전 행렬 (Rotation Matrix):")
        print(R)
        print("="*60)


def main():
    """
    메인 실행 함수
    
    사용법:
    1. 체커보드 캘리브레이션:
       python calibrate_camera.py --mode intrinsic --images ./calibration_images/*.jpg
    
    2. 아루코 마커 포즈 추정:
       python calibrate_camera.py --mode extrinsic --intrinsic ./camera_intrinsic.npz --image ./test_image.jpg
    
    권장 체커보드 패턴:
    - 다운로드: https://docs.opencv.org/4.x/pattern.png
    - 크기: 9x6 체커보드 (8x5 내부 코너)
    - 프린트 시 실제 크기 측정 후 square_size 조정 필요
    """
    parser = argparse.ArgumentParser(description='카메라 캘리브레이션 도구')
    parser.add_argument('--mode', choices=['intrinsic', 'extrinsic'], required=True,
                       help='캘리브레이션 모드 선택')
    
    # 내부 파라미터 캘리브레이션 옵션
    parser.add_argument('--images', type=str, help='체커보드 이미지들의 경로 (glob 패턴 사용 가능)')
    parser.add_argument('--checkerboard_size', nargs=2, type=int, default=[8, 5],
                       help='체커보드 내부 코너 개수 (가로, 세로)')
    parser.add_argument('--square_size', type=float, default=0.025,
                       help='체커보드 한 칸의 실제 크기 (미터)')
    
    # 외부 파라미터 추정 옵션
    parser.add_argument('--intrinsic', type=str, help='저장된 내부 파라미터 파일 경로')
    parser.add_argument('--image', type=str, help='아루코 마커가 있는 이미지 파일 경로')
    parser.add_argument('--marker_size', type=float, default=0.05,
                       help='아루코 마커의 실제 크기 (미터)')
    
    # 공통 옵션
    parser.add_argument('--visualize', action='store_true', help='결과 시각화')
    parser.add_argument('--output', type=str, help='결과 저장 경로')
    
    args = parser.parse_args()
    
    # 캘리브레이터 생성
    calibrator = CameraCalibrator(
        checkerboard_size=tuple(args.checkerboard_size),
        square_size=args.square_size
    )
    
    if args.mode == 'intrinsic':
        # 내부 파라미터 캘리브레이션
        if not args.images:
            print("Error: --images 옵션이 필요합니다.")
            return
        
        print("\n🎯 체커보드를 이용한 내부 파라미터 캘리브레이션")
        print("="*60)
        print(f"체커보드 크기: {args.checkerboard_size[0]}x{args.checkerboard_size[1]} 내부 코너")
        print(f"체커보드 칸 크기: {args.square_size}m")
        print("="*60)
        
        # 이미지 경로 수집
        image_paths = glob.glob(args.images)
        if not image_paths:
            print(f"Error: '{args.images}' 패턴에 해당하는 이미지를 찾을 수 없습니다.")
            return
        
        print(f"발견된 이미지: {len(image_paths)}개")
        
        # 캘리브레이션 데이터 수집
        successful_count = calibrator.collect_calibration_data(image_paths, args.visualize)
        
        if successful_count < 10:
            print(f"Warning: 성공한 이미지가 {successful_count}개로 적습니다. 최소 10개 이상 권장합니다.")
        
        if successful_count == 0:
            print("Error: 사용할 수 있는 이미지가 없습니다.")
            return
        
        # 첫 번째 이미지에서 이미지 크기 가져오기
        sample_image = cv2.imread(image_paths[0])
        image_shape = sample_image.shape[:2]
        
        # 캘리브레이션 수행
        reprojection_error, camera_matrix, dist_coeffs = calibrator.calibrate_intrinsic(image_shape)
        
        # 결과 출력
        calibrator.print_intrinsic_results(reprojection_error)
        
        # 결과 저장
        output_path = args.output or './camera_intrinsic.npz'
        calibrator.save_intrinsic_parameters(output_path)
        
    elif args.mode == 'extrinsic':
        # 외부 파라미터 추정
        if not args.intrinsic or not args.image:
            print("Error: --intrinsic 및 --image 옵션이 필요합니다.")
            return
        
        print("\n🎯 아루코 마커를 이용한 외부 파라미터 추정")
        print("="*60)
        print(f"마커 크기: {args.marker_size}m")
        print("="*60)
        
        # 내부 파라미터 로드
        calibrator.load_intrinsic_parameters(args.intrinsic)
        
        # 이미지 로드
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: 이미지를 읽을 수 없습니다: {args.image}")
            return
        
        # 아루코 마커 포즈 추정
        poses = calibrator.estimate_pose_single_marker(
            image, args.marker_size, visualize=args.visualize
        )
        
        if poses:
            print(f"\n검출된 마커 개수: {len(poses)}")
            for i, (rvec, tvec) in enumerate(poses):
                print(f"\n--- 마커 {i+1} ---")
                calibrator.print_extrinsic_results(rvec, tvec)
                
                # 결과 저장
                if args.output:
                    output_path = args.output.replace('.npz', f'_marker_{i}.npz')
                    calibrator.save_extrinsic_parameters(rvec, tvec, output_path)
        else:
            print("Error: 아루코 마커를 찾을 수 없습니다.")
        
        if args.visualize:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    📷 카메라 캘리브레이션 도구                    ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🔧 Step 1: 체커보드를 이용한 내부 파라미터 캘리브레이션
    📐 권장 체커보드 패턴: https://docs.opencv.org/4.x/pattern.png
    📏 체커보드 크기: 9x6 체커보드 (8x5 내부 코너)
    
    사용 예시:
    python calibrate_camera.py --mode intrinsic --images "./calib_images/*.jpg" --visualize
    
    🎯 Step 2: 아루코 마커를 이용한 외부 파라미터 추정
    
    사용 예시:
    python calibrate_camera.py --mode extrinsic --intrinsic "./camera_intrinsic.npz" --image "./aruco_test.jpg" --visualize
    """)
    
    main()