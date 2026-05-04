# CLIP Embeddings Derived from BABEL Text

This repository (`clip_enc_single/`, `pca/`) contains precomputed CLIP text embeddings based on a subset of action labels from the [BABEL dataset](https://babel.is.tue.mpg.de/).

## Contents

- `pca/clip_embeddings.tsv`: A `.tsv` file containing CLIP text embeddings computed from short action labels.
- `clip_enc_single/*.npy`: Frame-level CLIP text embeddings for each motion sequence in the overlapping subset of HumanML3D and BABEL.
- `clip_encoder.py`: Script used to process BABEL annotations and compute the above CLIP embeddings.
- HumanML3D subset lists:
  - Files ending with `_ft` correspond to sequences that exist in both HumanML3D and BABEL (i.e., have both sequence-level and frame-level text annotations).
  - Files ending with `_ft_no_overlap` are used to compare against models trained on the full BABEL dataset (e.g., FlowMDM-BABEL).
    - ⚠️ Note: Since HumanML3D and BABEL use different train/test splits, we have removed overlapping training samples of BABEL training set from the HumanML's test split to ensure a fair evaluation.

⚠️ **No original text or annotations from BABEL are included in this release**. All files are derived representations (CLIP features only).

## License

This release is for **non-commercial academic use only**, in accordance with the [BABEL dataset license](https://babel.is.tue.mpg.de/license.html). Users must obtain access to BABEL separately.

## Citation

If you use this data or related processing code, please cite the BABEL dataset:

```bibtex
@inproceedings{Punnakkal2021BABEL,
  title={BABEL: Bodies, Action and Behavior with English Labels},
  author={Punnakkal, Abhinanda R. and Ahmed, Umar Iqbal and Black, Michael J. and Gall, Jürgen and Varol, Gül},
  booktitle={CVPR},
  year={2021}
}
