# Human36M Multi-view Dataset for SelfPose3d

Human36M 데이터셋을 SelfPose3d 프레임워크에서 사용할 수 있도록 변환한 데이터셋입니다.

## 주요 특징

### 🎯 **다중 뷰 지원**
- Human36M의 4개 카메라 뷰를 모두 활용
- SelfPose3d의 Self-Supervised Learning 방식 적용
- 카메라 간 일관성을 통한 3D 포즈 추정

### 📊 **데이터 구성**
- **조인트 수**: 17개 (Human36M 표준)
- **카메라 수**: 4개 (모든 시점 사용)
- **대상자**: Protocol 2 기준 (S1, S5, S6, S7, S8 / S9, S11)
- **액션**: 15개 액션 타입

### 🔧 **기술적 특징**
- SelfPose3d의 `JointsDatasetSSV` 클래스 상속
- 3가지 증강 버전 자동 생성
- 다중 뷰 히트맵 및 3D 복셀 타겟 생성
- Human36M 표준 평가 메트릭 지원 (MPJPE, PA-MPJPE)

## 디렉토리 구조

```
data/Human36M/
├── images/                 # 이미지 파일들
├── annotations/           # JSON 어노테이션 파일들
│   ├── Human36M_subject*_data.json
│   ├── Human36M_subject*_camera.json
│   └── Human36M_subject*_joint_3d.json
└── noise_stats.py         # 노이즈 통계
```

## 설정 파일

```
configs/human36m/resnet50_human36m_4cam.yaml
```

### 주요 설정값
- **CAMERAS**: [0, 1, 2, 3] - 4개 카메라 모두 사용
- **NUM_JOINTS**: 17 - Human36M 조인트 수
- **BATCH_SIZE**: 2 - 메모리 효율성을 위해 작은 배치
- **MIN_VIEWS_CHECK**: 2 - 최소 2개 뷰 필요

## 사용법

### 1. 데이터셋 테스트
```bash
python test_human36m_dataset.py
```

### 2. 학습 실행
```bash
python tools/train_3d.py --cfg configs/human36m/resnet50_human36m_4cam.yaml
```

### 3. 평가 실행
```bash
python tools/validate_3d.py --cfg configs/human36m/resnet50_human36m_4cam.yaml \
                             --model-file output/human36m/model_best.pth.tar
```

## 클래스 구조

### `Human36MSSV` 클래스
- **부모 클래스**: `JointsDatasetSSV`
- **주요 메서드**:
  - `_get_db()`: Human36M 데이터를 SelfPose3d 형식으로 변환
  - `_get_cam()`: 카메라 파라미터 변환
  - `evaluate()`: Human36M 표준 평가

### 데이터 변환 과정
1. **프레임 그룹화**: 동일 프레임의 다중 뷰 데이터 그룹화
2. **카메라 파라미터 변환**: Human36M → SelfPose3d 형식
3. **3D→2D 프로젝션**: 카메라 내재 파라미터 사용
4. **다중 뷰 동기화**: 모든 카메라 뷰에서 일관된 데이터

## 평가 메트릭

### MPJPE (Mean Per Joint Position Error)
- 관절별 평균 위치 오차
- Human36M 표준 14개 관절 사용

### PA-MPJPE (Procrustes Aligned MPJPE)
- 강체 정렬 후 관절 위치 오차
- 포즈 형태 평가에 사용

### AP (Average Precision)
- 임계값별 정확도
- 25mm, 50mm, 75mm, 100mm, 125mm 임계값

## 주의사항

1. **메모리 사용량**: 4개 뷰 × 3개 증강 = 12배 메모리 사용
2. **배치 크기**: 큰 배치 크기 시 GPU 메모리 부족 가능
3. **최소 뷰 수**: 적어도 2개 뷰가 있는 프레임만 사용

## 향후 개선사항

- [ ] 동적 배치 크기 조절
- [ ] 카메라별 가중치 적용
- [ ] 시간적 일관성 활용
- [ ] SMPL 메시 복원 지원

## 참고

- **Human36M**: [공식 웹사이트](http://vision.imar.ro/human3.6m/description.php)
- **SelfPose3d**: [GitHub](https://github.com/yfeng95/SelfPose3d)
- **논문**: "Monocular 3D Multi-Person Pose Estimation by Integrating Top-Down and Bottom-Up Networks"
