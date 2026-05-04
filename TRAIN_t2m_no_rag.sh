#!/bin/bash

# No-RAG ablation launcher. Reuses TRAIN_t2m_rag.sh and forces DISABLE_RAG=1.
DISABLE_RAG=1 bash TRAIN_t2m_rag.sh "$@"
