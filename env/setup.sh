# package install
apt update && apt install htop vim unzip zip -y

# git
cp /workspace/env/tokens/.ssh/config /root/.ssh/config
cp /workspace/env/tokens/.ssh/id_rsa_git /root/.ssh/id_rsa_git
chmod 600 /root/.ssh/id_rsa_git
##  user setting
git config --global init.defaultBranch main
git config --global user.email "yuri620620@gmail.com"
git config --global user.name "YuriNakayama"
git config --global credential.helper store

# poetry install & setting
pip install poetry
poetry config virtualenvs.path --unset
poetry config virtualenvs.in-project true
## activate poetry and install library
poetry env use 3.11
poetry install
poetry shell
ipython kernel install --user --name=cate

#kaggle setup
mkdir -p /root/.kaggle
cp /workspace/env/tokens/kaggle.json /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json

# aws.envファイルを読み込み、環境変数として設定
mkdir -p /root/.aws
cat /workspace/env/tokens/aws.env > /root/.aws/credentials
chmod 600 /root/.aws/credentials

# hugging face setup
export HUGGINGFACE_TOKEN=$(cat /workspace/env/tokens/huggingface.env)
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

# wandb
# export WANDB_TOKEN=$(cat /workspace/env/tokens/wandb.env)
# wandb login $WANDB_TOKEN



source ~/.bashrc