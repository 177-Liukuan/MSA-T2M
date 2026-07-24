"""
Inspect shapes of motion latents saved by get_latent.py and get_msa_latent.py.

Usage:
    python inspect_latent_shapes.py        # auto-scan all known dirs
    python inspect_latent_shapes.py --n 5  # show first 5 samples per dir
"""

import os
import argparse
import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
BASE = os.path.join(REPO_ROOT, 'humanml3d_272')

DIRS = {
    # get_latent.py -> causal TAE latents
    'get_latent / causal_TAE': os.path.join(
        BASE, 't2m_latents', 'causal_TAE_t2m_272_h100_20260203'),

    # get_msa_latent.py -> local z_local latents
    'get_msa_latent / z_local [fulldb_right]': os.path.join(
        BASE, 't2m_latents_msa_vae',
        'MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right'),
    'get_msa_latent / z_local [main]': os.path.join(
        BASE, 't2m_latents_msa_vae',
        'MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048'),
    'get_msa_latent / z_local [no_local]': os.path.join(
        BASE, 't2m_latents_msa_vae',
        'MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_no_local'),
    'get_msa_latent / z_local [no_global]': os.path.join(
        BASE, 't2m_latents_msa_vae',
        'MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_no_global'),

    # get_msa_latent.py -> global h_cls features
    'get_msa_latent / h_cls [fulldb_right]': os.path.join(
        BASE, 'h_cls_latents_msa_vae',
        'MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right'),
    'get_msa_latent / h_cls [main]': os.path.join(
        BASE, 'h_cls_latents_msa_vae',
        'MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048'),
    'get_msa_latent / h_cls [no_local]': os.path.join(
        BASE, 'h_cls_latents_msa_vae',
        'MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_no_local'),
    'get_msa_latent / h_cls [no_global]': os.path.join(
        BASE, 'h_cls_latents_msa_vae',
        'MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_no_global'),
}


def load_npy_files(directory, n=5):
    if not os.path.isdir(directory):
        return None, '[DIR NOT FOUND]'
    files = sorted(f for f in os.listdir(directory) if f.endswith('.npy'))
    if not files:
        return None, '[EMPTY DIR]'
    samples = []
    for fname in files[:n]:
        arr = np.load(os.path.join(directory, fname), allow_pickle=False)
        samples.append((fname, arr))
    return samples, None


def inspect_dir(label, directory, n_sample=5):
    print(f'\n{"─"*70}')
    print(f'  {label}')
    print(f'  path: {directory}')
    print(f'{"─"*70}')

    samples, err = load_npy_files(directory, n=n_sample)
    if err:
        print(f'  {err}')
        return

    total = len([f for f in os.listdir(directory) if f.endswith('.npy')])
    print(f'  total .npy files : {total}')

    # separate reference files from regular samples
    reg_samples = [(f, a) for f, a in samples if not f.startswith('reference_end_latent')]
    ref_samples = [(f, a) for f, a in samples if f.startswith('reference_end_latent')]

    if reg_samples:
        print(f'  showing first    : {len(reg_samples)} regular samples')
        print()
        print(f'  {"filename":<45}  shape                dtype')
        print(f'  {"─"*45}  {"─"*20}  {"─"*8}')
        shapes = []
        for fname, arr in reg_samples:
            print(f'  {fname:<45}  {str(arr.shape):<20}  {arr.dtype}')
            shapes.append(arr.shape)

        # shape stats
        if len(shapes) > 1 and all(len(s) == len(shapes[0]) for s in shapes):
            print()
            print(f'  shape statistics over {len(shapes)} shown samples:')
            for dim in range(len(shapes[0])):
                vals = [s[dim] for s in shapes]
                print(f'    dim{dim}: min={min(vals)}, max={max(vals)}, '
                      f'mean={sum(vals)/len(vals):.1f}')

        # value range from first regular sample
        arr0 = reg_samples[0][1]
        print()
        print(f'  value range (first sample): '
              f'min={arr0.min():.4f}, max={arr0.max():.4f}, mean={arr0.mean():.4f}')

    # reference end latents
    ref_in_dir = [f for f in os.listdir(directory)
                  if f.startswith('reference_end_latent') and f.endswith('.npy')]
    if ref_in_dir:
        print()
        print(f'  reference end latent files inside this dir:')
        for rf in sorted(ref_in_dir):
            arr = np.load(os.path.join(directory, rf))
            print(f'    {rf}')
            print(f'      shape={arr.shape}  dtype={arr.dtype}  '
                  f'min={arr.min():.4f}  max={arr.max():.4f}')


def inspect_root_refs(root):
    ref_files = sorted(f for f in os.listdir(root)
                       if f.startswith('reference_end_latent') and f.endswith('.npy'))
    if not ref_files:
        return
    print(f'\n{"─"*70}')
    print(f'  Reference end latents (workspace root)')
    print(f'{"─"*70}')
    for rf in ref_files:
        arr = np.load(os.path.join(root, rf))
        print(f'  {rf}')
        print(f'    shape={arr.shape}  dtype={arr.dtype}  '
              f'min={arr.min():.4f}  max={arr.max():.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5,
                        help='Number of sample files to show per directory')
    args = parser.parse_args()

    root = REPO_ROOT

    print('=' * 70)
    print('  Motion Latent Shape Inspector')
    print('  Covers: get_latent.py  and  get_msa_latent.py')
    print('=' * 70)

    inspect_root_refs(root)

    for label, directory in DIRS.items():
        inspect_dir(label, directory, n_sample=args.n)

    print(f'\n{"=" * 70}')
    print('  Done.')
    print('=' * 70)


if __name__ == '__main__':
    main()
