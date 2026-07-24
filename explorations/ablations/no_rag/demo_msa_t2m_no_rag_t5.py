"""No-RAG ablation demo entry.

This wrapper reuses the demo pipeline and forces --disable_rag.
"""

import sys
from explorations.demos_and_diagnostics.demo_msa_t2m_t5 import main


if __name__ == '__main__':
    if '--disable_rag' not in sys.argv:
        sys.argv.append('--disable_rag')
    main()
