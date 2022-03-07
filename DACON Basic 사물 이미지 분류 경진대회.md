# DACON Basic 사물 이미지 분류 경진대회

대회 링크: https://dacon.io/competitions/official/235874/overview/description

순위 2등

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

*  script_sample paramter

  * script 폴더 안에 존재하는 .ini 파일
  * data_path: split된 train, test set을 포함하는 폴더의 주소
  * test_path: 제출용 test set
  * csv_path: sample_submission.csv 주소
  * output_dir: model이 저장될 폴더 이름
  * batch_size: 128
  * num_epoch: 2000
  * fold: 5
  * cutmix: 1
  * gpus: 사용할 gpu index
  * phase: train or test or inference
    * train: 학습
    * test: evaluation
    * inference: 제출용 output 제작
  * seed: random seed
  * lr: 0.01

* train

  * 학습시킬 데이터의 경로를 설정하고 output_dir에 모델이 저장될 폴더 이름을 설정 후 main.py 실행

    * ```shell
      python main.py script_sample
      ```

* test 및 inference

  * 대회 최종 제출 모델 주소

    * results/final_model

  * script_sample.ini 파일에서 최종 모델이 저장된 폴더의 이름을 output_dir에 설정하고 phase를 test or inference로 설정한 후 main.py 실행

    * ```shell
      python main.py script_sample
      ```

  * test 시 accuracy가 출력됨
  * inference 시 output_dir에 입력한 최종 모델이 저장된 폴더 내부에 final_output.csv 파일이 생성됨