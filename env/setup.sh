# package install
apt update && apt install htop vim unzip zip curl -y

# git
cp /workspace/env/tokens/.ssh/config /root/.ssh/config
cp /workspace/env/tokens/.ssh/id_rsa_git /root/.ssh/id_rsa_git
chmod 600 /root/.ssh/id_rsa_git
##  user setting
git config --global init.defaultBranch main
git config --global user.email "yuri620620@gmail.com"
git config --global user.name "YuriNakayama"
git config --global credential.helper store

# kaggle setup
mkdir -p /root/.config/kaggle
cp /workspace/env/tokens/kaggle.json /root/.config/kaggle/kaggle.json
chmod 600 /root/.config/kaggle/kaggle.json

# aws.envファイルを読み込み、環境変数として設定
mkdir -p /root/.aws
cp /workspace/env/tokens/aws.env /root/.aws/credentials
chmod 600 /root/.aws/credentials

# mlflow
mkdir -p /root/.mlflow
cp /workspace/env/tokens/mlflow.env /root/.mlflow/credentials
chmod 600 /root/.mlflow/credentials

# uv install & setting
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv sync --all-extras --dev
source .venv/bin/activate
ipython kernel install --user --name=cate

# hugging face setup
# export HUGGINGFACE_TOKEN=$(cat /workspace/env/tokens/huggingface.env)
# huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

# wandb
# export WANDB_TOKEN=$(cat /workspace/env/tokens/wandb.env)
# wandb login $WANDB_TOKEN

source ~/.bashrc