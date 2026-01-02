Chest X-ray Pneumonia Classification

PyTorch 기반 Chest X-ray 이진 분류 프로젝트 (NORMAL / PNEUMONIA)

ImageFolder 구조를 사용한 의료 영상 데이터 로딩

dataset.py / model.py / train.py로 역할 분리된 학습 파이프라인 구성

Train / Validation / Test 폴더 분리로 데이터 누수 방지

ResNet 계열 CNN 모델을 베이스라인으로 사용

사전 학습 가중치 기반 Fine-tuning 구조 적용

Epoch 단위 학습 및 검증 loss / accuracy 추적

Best model 기준 체크포인트 저장

Grad-CAM을 활용한 모델 예측 근거 시각화

로컬 학습 환경 기준 재현 가능한 구조로 설계
