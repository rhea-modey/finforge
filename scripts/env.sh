# Source this before running anything: `source scripts/env.sh`
set -a
source /Users/videetmehta/hack26/.env
source /Users/videetmehta/hack26/finforge/.env
ACTOR_API_KEY=${ACTOR_API_KEY:-$FIREWORKS_API_KEY}
OPTIMIZER_API_KEY=${OPTIMIZER_API_KEY:-$MINIMAX_API_KEY}
JUDGE_LLM_API_KEY=${JUDGE_LLM_API_KEY:-$FIREWORKS_API_KEY}
DATAGEN_API_KEY=${DATAGEN_API_KEY:-$OPENAI_API_KEY}
set +a
