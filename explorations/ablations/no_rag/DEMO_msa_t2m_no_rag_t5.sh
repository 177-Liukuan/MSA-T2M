#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

# Usage:
#   bash DEMO_msa_t2m_no_rag_t5.sh "a person waves both hands"

TEXT_PROMPT=${1:-"a person is walking in a circle"}

python -m explorations.ablations.no_rag.demo_msa_t2m_no_rag_t5 \
  --text "$TEXT_PROMPT" \
  --mode pos
