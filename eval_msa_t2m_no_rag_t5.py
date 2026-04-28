"""No-RAG ablation eval entry.

This wrapper reuses the main eval pipeline and forces --disable_rag.
"""

import sys
from eval_msa_t2m_rag_t5 import main


if __name__ == '__main__':
    if '--disable_rag' not in sys.argv:
        sys.argv.append('--disable_rag')
    main()
