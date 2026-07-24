#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

# No-RAG ablation launcher. Reuses TRAIN_t2m_rag.sh and forces DISABLE_RAG=1.
DISABLE_RAG=1 bash "$REPO_ROOT/TRAIN_t2m_rag.sh" "$@"
