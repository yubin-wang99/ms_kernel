# 새 서버 세팅 — 다운로드부터 실행, 그리고 Claude Code 설치

이 프로젝트(`ms_kernel`)를 **새 서버**에 받아 실행하는 전체 절차입니다. 단순히 conda/`nvcc`가 이미
갖춰진 서버로 **포팅**만 하려면 [`SETUP.md`](SETUP.md)를 보세요(거긴 toolchain이 있다고 가정).

**이 서버 전제:** GPU와 **CUDA가 이미 설치돼 있음** → **CUDA를 새로 깔거나 다운그레이드하지 않습니다.**
대신 **기존 CUDA 버전을 확인하고, 그 버전에 맞는 PyTorch wheel을 설치**합니다.

**개발/검증 스택(참고):** Ubuntu · Python 3.10 · PyTorch 2.5.1 · NVIDIA **RTX 3090 (sm_86)**
(원래는 CUDA 11.8 + cu118 torch로 빌드/검증했지만, **서버의 기존 CUDA에 맞추면 됩니다**).

**레포(private):** `git@github.com:yubin-wang99/ms_kernel.git`

> 자주 막히는 3가지:
> 1. **빌드된 `.so`는 머신 종속** — 커밋된 빌드 산출물을 재사용하지 말고, 타깃에서 `setup.py build_ext`로
>    **새로 빌드**하세요.
> 2. **GPU 아키텍처** — RTX 3090(sm_86)이 아니면 `setup.py`의 arch 플래그를 수정
>    (A100=`sm_80`, RTX 4090=`sm_89`, H100=`sm_90`).
> 3. **torch의 CUDA = 시스템 CUDA 일치** — 시스템 `nvcc` 버전에 맞는 torch wheel을 깔아야 빌드가 됩니다.

---

## 0. GPU · 드라이버 · CUDA 확인 (이미 있음 — 설치 X)

```bash
nvidia-smi          # GPU + 드라이버 보이는지
nvcc --version      # ★ 기존 CUDA 버전 확인 (예: release 12.1) — 이 버전을 그대로 사용, 다운그레이드 금지
python3 -c "import torch; print(torch.cuda.get_device_capability(0))" 2>/dev/null  # GPU arch 확인용(torch 있으면)
```
`nvcc`가 PATH에 없고 `/usr/local/cuda*/bin`에만 있으면 PATH만 잡아주세요(설치/변경 아님):
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 1. 시스템 패키지 (git, 빌드 도구, python)

```bash
sudo apt update
sudo apt install -y git build-essential python3.10 python3.10-venv python3-pip
```

## 2. GitHub 인증 + 클론 (private 레포)

**SSH 키 (권장):**
```bash
ssh-keygen -t ed25519 -C "your_email"      # 프롬프트는 엔터 연타
cat ~/.ssh/id_ed25519.pub                  # 출력된 키를 GitHub > Settings > SSH and GPG keys 에 등록
ssh -T git@github.com                       # "Hi <user>! ..." 나오면 성공
git clone git@github.com:yubin-wang99/ms_kernel.git
cd ms_kernel
```
**HTTPS + 토큰 (대안):** `git clone https://github.com/yubin-wang99/ms_kernel.git` 후, 비밀번호 대신
Personal Access Token(github.com/settings/tokens) 입력.

## 3. Python 환경 + 의존성 + CUDA 확장 빌드

**핵심: 서버의 기존 CUDA 버전에 맞는 torch wheel을 설치** (CUDA를 건드리지 않음).
`nvcc --version`에서 본 버전에 맞춰 index-url만 고르세요:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 기존 CUDA에 맞는 torch 설치 — 아래 중 서버 CUDA에 해당하는 한 줄만:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
# pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 # CUDA 12.1
# pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 # CUDA 12.4
# (시스템 CUDA가 위에 없으면, torch 버전을 그 CUDA를 지원하는 것으로 맞추세요. 핵심은 'torch CUDA = 시스템 nvcc')

pip install -r requirements.txt
# precision / E2E 스크립트용 추가 패키지 (requirements.txt엔 torch/numpy/ninja/pytest만 있음):
pip install transformers datasets safetensors tqdm pandas matplotlib seaborn scipy
/
# CUDA 확장 빌드 → setup.py 옆에 ms_cuda*.so 생성
python setup.py build_ext --inplace

# 빌드 확인 (torch.ops.msaq.* 등록됨)
python -c "import ms_cuda, torch; print('ok:', [o for o in dir(torch.ops.msaq)][:5])"
```

> GPU가 **sm_86이 아니면** 빌드 전에 `setup.py`의 `nvcc_flags` arch를 수정하세요(상단 주의 / `SETUP.md` §4).
> 확인: `python -c "import torch; print(torch.cuda.get_device_capability(0))"`.

## 4. 모델 — Llama-3.1-8B-Instruct (PPL/E2E 스크립트가 사용)

게이트된 HF 모델이라 토큰 + 라이선스 동의가 필요합니다. 스크립트는
`~/.cache/huggingface/hub/...Llama-3.1-8B-Instruct` 경로를 찾습니다(예: `precision/lightms_qsnr.py`).

```bash
pip install huggingface_hub
huggingface-cli login          # HF 토큰 입력 (huggingface.co/settings/tokens)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

## 5. 실행

```bash
# 커널 마이크로벤치
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/benchmark.py
# 정확도(PPL) 예시
CUDA_VISIBLE_DEVICES=0 python precision/lightms_qsnr.py
# E2E 배치 스윕
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/harness_batchsweep.py
```

---

## 6. Claude Code 설치

Node.js 18+ 필요.

```bash
# nvm로 Node 설치 (권장)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc
nvm install --lts

# Claude Code 설치
npm install -g @anthropic-ai/claude-code
cd ~/ms_kernel
claude                  # 첫 실행 시 인터랙티브 로그인
```

> 로그인은 인터랙티브입니다. Claude Code 세션 안에서 직접 실행하려면 `!` 접두사(`! claude`)로 돌리면
> 출력이 대화에 캡처됩니다.

---

## 트러블슈팅 체크리스트

- `nvcc: command not found` → 기존 CUDA의 `bin`을 PATH에 추가(§0). **재설치/다운그레이드 아님.**
- 빌드 시 arch 에러 / 런타임 `no kernel image` → GPU가 sm_86이 아님 → `setup.py` arch 플래그 수정.
- import 시 `RuntimeError: CUDA error` → torch의 CUDA 버전 ≠ 시스템 CUDA → §3에서 시스템 CUDA에 맞는
  torch wheel로 재설치.
- HF `401 / gated` → 로그인 안 됐거나 라이선스 미동의(§4).
- 빌드가 느림 → `pip install ninja` (없으면 setup.py가 느린 distutils로 폴백).
