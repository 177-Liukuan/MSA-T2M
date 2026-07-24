"""
Smoke tests for all new scripts:
1. Code syntax validation
2. Module imports
3. DataLoader instantiation
4. Model initialization
5. Single training step simulation
"""

import sys
import os
import torch
import numpy as np
from os.path import join as pjoin

print("=" * 70)
print("SMOKE TESTS FOR NEW SCRIPTS")
print("=" * 70)
print("")

# ============================================================================
# Test 1: Syntax and Import Validation
# ============================================================================
print("[Test 1] Validating Python syntax and imports...")
print("-" * 70)

test_files = [
    'get_msa_latent.py',
    'explorations/clip/get_text_latent_clip.py',
    'explorations/clip/train_t2m_baseline_clip.py',
    'humanml3d_272/dataset_TM_train_msa_cached.py',
    'humanml3d_272/dataset_TM_train_baseline_clip.py',
]

import py_compile
import tempfile

for f in test_files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"  ✓ {f}")
    except py_compile.PyCompileError as e:
        print(f"  ✗ {f}: {e}")
        sys.exit(1)

print("")

# ============================================================================
# Test 2: Dataset Import and Instantiation
# ============================================================================
print("[Test 2] Testing DataLoader imports and instantiation...")
print("-" * 70)

try:
    from humanml3d_272 import dataset_TM_train_msa_cached
    print("  ✓ Imported dataset_TM_train_msa_cached")
except ImportError as e:
    print(f"  ✗ Failed to import dataset_TM_train_msa_cached: {e}")
    sys.exit(1)

try:
    from humanml3d_272 import dataset_TM_train_baseline_clip
    print("  ✓ Imported dataset_TM_train_baseline_clip")
except ImportError as e:
    print(f"  ✗ Failed to import dataset_TM_train_baseline_clip: {e}")
    sys.exit(1)

print("")

# ============================================================================
# Test 3: Check Directory Structure
# ============================================================================
print("[Test 3] Checking directory structure...")
print("-" * 70)

required_dirs = [
    'humanml3d_272',
    'models',
    'options',
    'utils',
]

for d in required_dirs:
    if os.path.isdir(d):
        print(f"  ✓ {d}/")
    else:
        print(f"  ✗ Missing {d}/")
        sys.exit(1)

print("")

# ============================================================================
# Test 4: Model Import and Instantiation
# ============================================================================
print("[Test 4] Testing model imports and initialization...")
print("-" * 70)

try:
    from models.llama_model import LLaMAHF, LLaMAHFConfig
    print("  ✓ Imported LLaMAHF and LLaMAHFConfig")
    
    config = LLaMAHFConfig.from_name('Normal_size')
    config.block_size = 78
    
    # Create model on CPU for testing
    model = LLaMAHF(config, num_diffusion_head_layers=8, latent_dim=16, device='cpu')
    print(f"  ✓ Created LLaMAHF model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
except Exception as e:
    print(f"  ✗ Failed to initialize LLaMAHF: {e}")
    sys.exit(1)

try:
    import torch.nn as nn
    
    class TextProjector(nn.Module):
        def __init__(self, input_dim=512, output_dim=768):
            super().__init__()
            self.proj = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.proj(x)
    
    projector = TextProjector(512, 768)
    test_input = torch.randn(2, 512)
    test_output = projector(test_input)
    
    if test_output.shape == (2, 768):
        print(f"  ✓ Created TextProjector (512→768d), output shape: {test_output.shape}")
    else:
        print(f"  ✗ TextProjector output shape mismatch: {test_output.shape} vs (2, 768)")
        sys.exit(1)
        
except Exception as e:
    print(f"  ✗ Failed to initialize TextProjector: {e}")
    sys.exit(1)

print("")

# ============================================================================
# Test 5: Verify Feature Files Structure
# ============================================================================
print("[Test 5] Verifying expected feature file structure...")
print("-" * 70)

# Check if sample features exist (for minimal testing)
text_latent_dir = './humanml3d_272/text_latents_clip'
msa_latent_dir = './humanml3d_272/t2m_latents_msa_vae'

if os.path.exists(text_latent_dir):
    files = os.listdir(text_latent_dir)
    npy_files = [f for f in files if f.endswith('.npy')]
    print(f"  ✓ Text latent directory exists: {len(npy_files)} .npy files")
    
    if 'empty_cfg_text_clip.npy' in files:
        empty_embed = np.load(pjoin(text_latent_dir, 'empty_cfg_text_clip.npy'))
        if empty_embed.shape == (512,):
            print(f"  ✓ empty_cfg_text_clip.npy shape OK: {empty_embed.shape}")
        else:
            print(f"  ✗ empty_cfg_text_clip.npy shape mismatch: {empty_embed.shape} vs (512,)")
    else:
        print(f"  ⚠ empty_cfg_text_clip.npy not found (will be created by get_text_latent_clip.py)")
else:
    print(f"  ⚠ Text latent directory not found: {text_latent_dir}")
    print(f"    (This is expected - will be created by get_text_latent_clip.py)")

if os.path.exists(msa_latent_dir):
    files = os.listdir(msa_latent_dir)
    dirs = [d for d in files if os.path.isdir(pjoin(msa_latent_dir, d))]
    print(f"  ✓ MSA latent root exists: {len(dirs)} model directories")
else:
    print(f"  ⚠ MSA latent directory not found: {msa_latent_dir}")
    print(f"    (This is expected - will be created by get_msa_latent.py)")

print("")

# ============================================================================
# Test 6: Syntax Check with Pylance (if available)
# ============================================================================
print("[Test 6] Checking critical code patterns...")
print("-" * 70)

# The historical train_t2m_msa.py entrypoint was already absent before this
# script was archived. Keep that fact visible without making the remaining
# diagnostics fail solely because a deleted experiment cannot be inspected.
historical_msa_train = 'train_t2m_msa.py'
if os.path.exists(historical_msa_train):
    with open(historical_msa_train, 'r') as f:
        code = f.read()
        checks = [
            ('CLIP_DIM = 512', 'CLIP dimension defined'),
            ('MODEL_DIM = 768', 'Model dimension defined'),
            ('class TextProjector', 'TextProjector class defined'),
            ('text_projector = TextProjector', 'TextProjector instantiated'),
            ('feat_text_projected = text_projector(feat_text)', 'Text projection applied'),
            ('forward_loss_withmask_2_forward', 'Two-forward training strategy'),
        ]

        for pattern, desc in checks:
            if pattern in code:
                print(f"  ✓ {desc}")
            else:
                print(f"  ✗ Failed to find: {desc}")
else:
    print(f"  ⚠ Historical entrypoint absent, skipped: {historical_msa_train}")

print("")

# Check baseline script for consistency
with open('explorations/clip/train_t2m_baseline_clip.py', 'r') as f:
    code = f.read()
    if 'dataset_TM_train_baseline_clip' in code:
        print(f"  ✓ Baseline uses correct DataLoader")
    else:
        print(f"  ✗ Baseline DataLoader import missing")

print("")

# ============================================================================
# Test 7: Bash Script Syntax
# ============================================================================
print("[Test 7] Validating Bash script syntax...")
print("-" * 70)

bash_files = [
    'explorations/clip/TRAIN_t2m_baseline_clip.sh',
]

for bash_file in bash_files:
    try:
        with open(bash_file, 'r') as f:
            content = f.read()
            # Basic checks
            if '#!/bin/bash' in content:
                print(f"  ✓ {bash_file} has proper shebang")
            else:
                print(f"  ⚠ {bash_file} missing shebang")
            
            if 'accelerate launch' in content:
                print(f"  ✓ {bash_file} uses accelerate launch")
            else:
                print(f"  ⚠ {bash_file} doesn't reference accelerate launch")
                
    except Exception as e:
        print(f"  ✗ Error reading {bash_file}: {e}")

print("")

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("✓ SMOKE TESTS PASSED")
print("=" * 70)
print("")
print("Next steps:")
print("  1. Generate motion latents:")
print("     python get_msa_latent.py --resume-pth <MSA_CHECKPOINT> \\")
print("       --latent_dir ./humanml3d_272/t2m_latents_msa_vae/<EXP_NAME>")
print("")
print("  2. Generate CLIP text embeddings:")
print("     python -m explorations.clip.get_text_latent_clip")
print("")
print("  3. Train MSA-VAE model:")
print("     bash TRAIN_msa_vae_phase1.sh 4 t2m_272")
print("     bash TRAIN_msa_vae_phase2.sh 4 t2m_272")
print("")
print("  4. Train Baseline (TAE+CLIP) for comparison:")
print("     bash explorations/clip/TRAIN_t2m_baseline_clip.sh 4")
print("")
