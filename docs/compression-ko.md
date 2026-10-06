# 체크포인트 용량 축소·학생 모델 지식 증류 사용법

모든 명령은 저장소 루트에서 실행합니다. 실제 학습은 GPU가 있는 별도
환경에서 실행하세요. 이 브랜치에는 구현과 단위 검증 결과가 있으며,
학습된 학생 모델이나 정확도·속도 측정 결과는 아직 없습니다.

## 1. 브랜치와 실행 환경 준비

기존 checkout에서 브랜치를 가져옵니다. checkout이 없으면 먼저 이 저장소를
clone하세요. 작업 중인 변경 사항은 브랜치 전환 전에 보존합니다.

```bash
git fetch origin refs/heads/feat/checkpoint-export-distillation:refs/remotes/origin/feat/checkpoint-export-distillation
git switch feat/checkpoint-export-distillation
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
lithium-libs artifacts
```

Python 3.12와 PyTorch 2.11.0을 사용합니다. GPU 환경에는 해당 장치·드라이버에
맞는 CUDA 빌드를 공식 PyTorch 설치 방법으로 준비하세요. CPU 빌드가 이미
설치돼 있으면 CUDA 빌드로 교체해야 합니다. 다음 명령으로 확인합니다.

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

CUDA 학습 환경에서는 두 번째 값이 `True`여야 합니다.

## 2. optimizer를 제거한 교사 파일 만들기

```bash
mkdir -p outputs/training outputs/reports
lithium-libs export-checkpoint \
  --output outputs/training/teacher_inference.pth \
  > outputs/reports/teacher_export.json
```

명령이 성공하면 원본 `solution_selector.pth`의 SHA-256을 확인하고,
optimizer·scheduler·학습 이력 없이 모델 설정과 가중치·버퍼만 저장합니다.
JSON 보고서에는 파일 크기, 원본/출력 SHA-256, 파라미터 수와
`tensors_identical: true`가 기록됩니다. 출력 파일이 이미 있으면 거부하므로
반복 실행할 때는 새 파일명을 쓰거나 검증된 기존 파일을 재사용하세요.

다음 명령은 성공한 export의 해시를 읽습니다. 이후 예제는 같은 셸에서
이 변수를 사용합니다. 셸을 새로 열면 다시 설정하세요.

```bash
teacher_sha256="$(python -c 'import json; print(json.load(open("outputs/reports/teacher_export.json"))["exported_sha256"])')"
```

원본 파라미터 수·추론 연산량은 그대로입니다. 이 파일은 추론 또는 교사로
사용할 수 있지만 기존 Adam 상태를 이어받는 학습 재개용 파일은 아닙니다.

## 3. 학생 설정 확인 — 학습 없음

기본 설정 파일은 `configs/distillation_small.json`입니다.

```bash
lithium-libs distill \
  --teacher outputs/training/teacher_inference.pth \
  --expected-sha256 "$teacher_sha256" \
  --config configs/distillation_small.json \
  --device cuda \
  --output-dir outputs/training/student_small \
  --dry-run
```

`--dry-run`은 설정과 총 Adam step 수만 확인합니다. 모델·체크포인트 로딩,
데이터 생성, 학습, 출력 디렉터리 생성은 하지 않습니다. CUDA 연결·메모리
용량·교사 파일의 실제 해시는 학습을 시작할 때 확인됩니다.

기본 학생은 **폭 144 / 3개 블록 / 8 heads / MLP 512·256**,
총 **2,664,722개 파라미터**입니다. 입력 두 채널·512점, 출력 8행·300 bins와
물리 decoder는 유지합니다. 교사 모델은 eval 상태로 고정하고 역전파하지
않습니다. 원래 7항 물리 손실에 전체 logits KL, shift, 재구성 증류 손실을
더합니다. 상세 수식은 [영문 구현 문서](compression.md)에 있습니다.

## 4. GPU 환경에서 짧은 확인 실행

처음에는 1회 update로 실행 경로와 메모리 사용을 확인하는 것을 권장합니다.
기본 설정을 복사해 별도 JSON을 만듭니다. 아래 파일 생성은 기존 파일을
덮어쓰지 않으며, 이미 있으면 새 이름을 쓰거나 직접 수정하세요.

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("configs/distillation_small.json").read_text())
config["student"].update(
    online_updates=1,
    samples_per_online_update=32,
    batch_size=32,
    validation_samples=32,
    number_of_workers=0,
)
with Path("outputs/training/distillation_check.json").open("x") as stream:
    json.dump(config, stream, indent=2)
PY

lithium-libs distill \
  --teacher outputs/training/teacher_inference.pth \
  --expected-sha256 "$teacher_sha256" \
  --config outputs/training/distillation_check.json \
  --device cuda \
  --output-dir outputs/training/student_check
```

**이 명령은 실제 데이터 생성과 학습을 수행합니다.** 이번 구현 작업에서는
실행하지 않았습니다. 출력 디렉터리는 새로 만들거나 비어 있어야 합니다.

## 5. 본 학습 실행

짧은 실행을 확인한 뒤 별도 출력 디렉터리에서 시작합니다.

```bash
lithium-libs distill \
  --teacher outputs/training/teacher_inference.pth \
  --expected-sha256 "$teacher_sha256" \
  --config configs/distillation_small.json \
  --device cuda \
  --output-dir outputs/training/student_small
```

기본값은 batch 32, update당 새 스펙트럼 512개, 20,000 updates입니다.
총 320,000 Adam steps와 10,240,000 학습 스펙트럼이 생성됩니다. 실험 시작
설정이며, 목표 장치에서 필요한 시간·메모리나 정확도는 측정하지 않았습니다.
오래 실행할 때는 GPU 환경의 작업 스케줄러나 터미널 유지 기능을 사용하세요.

설정을 바꿀 때는 JSON을 별도 파일로 복사하고 `--config`로 지정합니다.

| JSON의 `student` 항목 | 용도·조건 |
|---|---|
| `base_dimension` | 임베딩 폭. 양의 짝수이며 heads로 나누어져야 함. 보수적 후보는 192 |
| `number_of_attention_blocks` | 블록 수. 기본 3 |
| `number_of_attention_heads` | heads 수. 기본 8; 이것만 줄이면 가중치 크기가 줄지 않음 |
| `classifier_hidden_dimensions` | MLP 두 hidden 폭. 기본 `[512, 256]` |
| `batch_size` | GPU 메모리에 맞춰 조절 |
| `samples_per_online_update` | batch 이상이며 batch로 나누어져야 함 |
| `online_updates` | 전체 학습량. Adam steps = updates × samples/update ÷ batch |
| `validation_samples` | 실행 시작 시 생성해 고정할 synthetic 검증셋 크기 |
| `number_of_workers` | CPU 데이터 생성 worker 수. 문제 진단 시 0 |
| `learning_rate`, `minimum_learning_rate` | Adam 및 cosine schedule 설정 |
| `random_seed` | 새 실행의 seed. 기본 42 |

`temperature`, `supervised_weight`, `logits_weight`, `shift_weight`,
`reconstruction_weight`, `components_weight`는 JSON 최상위 항목입니다.
기본 온도는 2, 가중치는 components만 0이고 나머지는 1입니다.
물리 supervision은 양수로 유지해야 합니다.

## 6. 결과 파일과 학생 모델 사용

| 파일 | 용도 |
|---|---|
| `distillation_config.json` | 실제 실행 설정과 교사 SHA-256 |
| `distillation_history.csv` | 학습·검증 손실, 동위원소 MAE, 시간·학습률 |
| `best_model.pth` | 가장 좋은 학생의 가중치·설정·optimizer·scheduler·metrics |
| `best_model_inference.pth` | **배포·추론용 학생 파일**, optimizer 없음 |
| `checkpoint_latest.pth` | 기본 10 updates 간격의 학습 상태 |
| `checkpoint_update_N.pth` | 기본 1,000 updates 간격의 영구 학습 상태 |

best는 교사와의 일치도만으로 고르지 않고 **물리 supervision 검증 손실**로
선택합니다. 현재 CLI는 매번 새 학습을 시작하며 자동 resume는 지원하지
않습니다. 중단 후 같은 디렉터리에서 명령을 다시 실행하면 거부됩니다.

추론용 학생 파일에는 크기 정보가 포함돼 있어 별도 architecture 인자 없이
불러올 수 있습니다.

```python
import numpy as np
import torch
from lithium_libs_solution_selection.model import load_checkpoint

model = load_checkpoint(
    "outputs/training/student_small/best_model_inference.pth",
    device="cuda",
)
# 기존 전처리를 마친 Li·Ne 스펙트럼: float32 [N, 2, 512].
inputs = torch.from_numpy(np.load("spectra.npy", allow_pickle=False)).float().to("cuda")
with torch.inference_mode():
    components, isotope_logits, shift_px, reconstruction = model(inputs)
    isotope_percentage = (
        isotope_logits.softmax(dim=-1) * model.get_predictvalue.bin_centers[6]
    ).sum(dim=-1)
```

## 문제 해결·평가

- CUDA 미인식: GPU 할당, 드라이버, PyTorch CUDA 빌드를 확인합니다.
- GPU 메모리 부족: batch를 낮추고 samples/update의 배수 조건을 유지합니다.
  이 구현은 FP32이며 AMP를 사용하지 않습니다.
- worker 오류: `number_of_workers=0`으로 짧은 실행을 확인합니다.
- 해시 오류: export 보고서의 `exported_sha256`과 실제 교사 파일이 대응하는지
  확인합니다. 원본 manifest나 검증 해시를 바꿔 통과시키지 않습니다.
- 출력 경로 오류: 실험별 새 디렉터리를 사용합니다. 원본 artifacts 안에는
  저장하지 않습니다.

단위 검증 명령은 `python -m pytest tests/test_compression.py -q`입니다.
이는 실제 증류 학습이나 CUDA 동작 검증을 대신하지 않습니다. 학습 후에는
독립 synthetic 데이터와 실측 데이터에서 동위원소·shift·선폭·성분 분해·
재구성 오차를 평가하고 목표 장치의 메모리와 latency를 측정하세요.
공개 `verify-inference`는 원본 체크포인트 전용이며, frozen 그림 재현은
학생 모델의 정확도 평가가 아닙니다.
