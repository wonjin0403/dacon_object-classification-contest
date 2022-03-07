# DACON Basic 사물 이미지 분류 경진대회

대회 링크: https://dacon.io/competitions/official/235874/overview/description

순위 2등

## 학습 방법 및 실험 내용

1. 모델
   ResNet18, Efficientnet-b0, ViT, ConvNext 모델을 사용하여 실험을 진행했고, 최종적으로 ResNet18을 선택했습니다. CIFAR10 SOTA 모델이 ViT‑H/14모델인 것을 확인하고 ViT를 시도해 보았으나 ResNet18과 유사한 성능을 보였습니다. Efficientnet-b0 모델도 ResNet18과 큰 차이를 보이지 않았고 ConvNext는 모델 학습시간이 너무 오래 걸려 끝까지 학습을 진행하지 못했습니다. 결과적으로 성능에 큰 차이가 없고 학습시간이 빠른 ResNet18을 finetuning 하는 방향으로 실험을 진행하게 되었습니다.

2. 데이터

   - train 데이터의 80%를 학습용으로, 20%를 test 용으로 사용했습니다.

   - Resize(128 x 128): ResNet18에 있어서 128보다 큰 size로 Resize 했을 때와 차이가 없어 성능을 유지하는 가장 작은 size를 선택했습니다.

   - Augmentation

     - RandomCrop(32, padding=4)
     - HorizontalFlip: Vertical flip과 함께 사용할 때보다 Horizontal flip만 사용할 때 acc가 더 향상되었습니다.
     - CIFAR10 Autoaugmentation: augmentation을 통해 90%에서 91%까지 acc 상승효과를 얻을 수 있었습니다. 외부 데이터를 사용하거나 pretrained 된 모델을 사용한 것이 아니라 논문에 나와있는 augmentation 종류와 수치만을 이용했습니다. 
       관련눈문(AutoAugment: Learning Augmentation Strategies from Data)
       코드는 "https://github.com/DeepVoltaire/AutoAugment"를 참고했습니다.

     * Cutout(n_holes=1, length=16): Cutout을 통해 91%에서 92%까지 acc가 상승하는 효과를 얻을 수 있었습니다. length 값은 “Improved Regularization of Convolutional Neural Networks with Cutout” 논문을 참고하여 정했습니다. 논문에 내용을 보면 CIFAR-10에 대해서 16 x 16 size로 cutout 한 것을 확인할 수 있고 이를 통해 성능이 향상된 것을 확인할 수 있습니다.

     * CutMix: 성능 향상에는 효과가 없었지만 학습 속도를 더 빠르게 해주었습니다.

3. 학습

   - optimizer로는 SGD를 사용했습니다. adam, Radam, adamW 등을 사용해보았으나 SGD가 가장 좋은 성능을 보였습니다.

   - 스케줄러로는 ReduceLROnPlateau를 사용했습니다. min_lr가 일정 이상 떨어지면 더 이상 성능이 향상되지 않아 min_lr=0.000001로 설정했습니다.

   - 5 fold로 학습을 진행했습니다.

4. 추론

   - Test set에 대해서 hard voting보다 soft voting 이 더 좋은 성능 보였기 때문에 soft voting 방식을 이용했습니다. 

   - TTA: HorizontalFlip, Rotate90(angles=[0, 30])만을 이용했습니다.

   - k fold와 TTA를 통해 92%에서 93%이상으로 성능이 향상되었습니다.

## 개발환경

"Ubuntu 16.04.6 LTS"

## 라이브러리 버전

python==3.6.13

timm==0.5.4

numpy==1.16.4

opencv-python==4.5.4-dev

pandas==1.1.5

tensorboard==2.7.0

scikit-learn==0.24.2

pytorch==1.4.0+cu100

tqdm==4.62.3

## pip

```shell
pip install ttach
```

## 실행 방법

```shell
python main.py script_sample
```

## script_sample parameter 설정

script 폴더 안에 존재하는 .ini 파일로 학습에 필요한 parameter를 설정

*  data_path: split된 train, test set을 포함하는 폴더의 주소
*  test_path: 제출용 test set
*  csv_path: sample_submission.csv 주소
*  output_dir: model이 저장될 폴더 이름
*  batch_size: 128
*  num_epoch: 2000
*  fold: 5
*  cutmix: 1
*  gpus: 사용할 gpu index
*  phase: train or test or inference
  * train: 학습
  * test: evaluation
  * inference: 제출용 output 제작
*  seed: random seed
*  lr: 0.01

## 데이터 unzip

* data 폴더에 있는 new_train.zip과 new_test.zip을 unzip

  ```shell
  cd ./data
  unzip new_train.zip
  unzip new_test.zip
  cd ..
  ```

## Train

* 학습시킬 데이터의 경로를 설정하고 output_dir에 모델이 저장될 폴더 이름을 설정 후 main.py 실행

  ```shell
  python main.py script_sample
  ```

## Test 및 Inference

* 대회 최종 제출 모델 주소

  * results/final_model

* script_sample.ini 파일에서 최종 모델이 저장된 폴더의 이름을 output_dir에 설정하고 phase를 test or inference로 설정한 후 main.py 실행

  * ```shell
    python main.py script_sample
    ```

* test 시 accuracy가 출력됨

* inference 시 output_dir에 입력한 최종 모델이 저장된 폴더 내부에 final_output.csv 파일이 생성됨