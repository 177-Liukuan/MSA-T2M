#!/bin/bash

# Usage:
#   bash DEMO_msa_t2m_no_rag_t5.sh "a person waves both hands"

TEXT_PROMPT=${1:-"a person is walking in a circle"}

python demo_msa_t2m_no_rag_t5.py \
  --text "$TEXT_PROMPT" \
  --mode pos
