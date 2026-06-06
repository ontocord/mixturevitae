#!/usr/bin/env bash
# TODO set those as arguments
if [ $# -lt 2 ]; then
    echo "Error: Not enough arguments provided."
    echo "Usage: $0 VENV_DIR HF_HOME"
    exit 1
fi

export VENV_DIR=$1
export HF_HOME=$2

RED='\033[0;31m'
NC='\033[0m'

echo -e "${RED}Virtual Environment Directory: $VENV_DIR${NC}"
echo -e "${RED}Hugging Face Home Directory: $HF_HOME${NC}"

if [ -d $VENV_DIR ]; then
  echo -e "${RED}Found existing venv at $VENV_DIR activating it.${NC}"
  source $VENV_DIR/bin/activate
else
  if ! command -v uv &> /dev/null; then
    echo -e "${RED}Installing uv${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  
  echo -e "${RED}Creating venv${NC}"
  uv venv $VENV_DIR --python 3.12 --managed-python
  source $VENV_DIR/bin/activate

  echo -e "${RED}Installing harness accelerate and dependencies${NC}"
  pushd $VENV_DIR
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
  pushd lm-evaluation-harness
  uv pip install -e . datasets==2.16.0 accelerate sentencepiece wandb
fi

echo -e "${RED}Download datasets${NC}"
mkdir -p $HF_HOME
TASKS="commonsense_qa,piqa,winogrande,arc_challenge,arc_easy,mmlu,mmlu_continuation,hellaswag,copa,openbookqa,lambada_openai,winogrande,boolq,social_iqa,mmlu_pro"

# Run a tiny model on all datasets to make sure they are all available locally
# A cleaner approach would to download the dataset directly
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float," \
    --tasks $TASKS \
    --batch_size 1 \
    --limit 1 \
    --trust_remote_code \
    --device cpu
