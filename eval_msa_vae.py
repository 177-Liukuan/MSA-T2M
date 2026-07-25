"""Standalone MSA-VAE evaluation for HumanML3D or the BABEL stream."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.tensorboard import SummaryWriter

import models.msa_vae as msa_vae
import options.option_msa_vae as option_msa_vae
import utils.eval_trans as eval_trans
import utils.utils_model as utils_model
from utils.eval_msa_vae_babel import (
    evaluate_msa_vae_babel,
    prepare_babel_validation_loader,
    validate_msa_checkpoint_metadata,
)


class EvalCompat(nn.Module):
    """Adapt the dict MSA-VAE output to legacy HumanML evaluation."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, motion):
        output = self.model(motion)
        return output["x_recon"], output["mu"], output["logvar"]


def build_validation_loader(
    args,
    humanml_loader_factory=None,
    babel_loader_factory=None,
):
    """Build only the loader belonging to the selected evaluation domain."""
    if args.msa_data_mode == "babel_sparse_global":
        if babel_loader_factory is None:
            from humanml3d_272.dataset_msa_vae_babel import ValidationDATALoader

            babel_loader_factory = ValidationDATALoader
        return babel_loader_factory(
            batch_size=args.batch_size,
            babel_motion_dir=args.babel_val_motion_dir,
            babel_text_dir=args.babel_val_text_dir,
            babel_cache_dir=args.babel_val_t5_cache_dir,
            babel_cache_manifest=args.babel_val_cache_manifest,
            babel_split="val",
            t5_model_path=args.t5_model_path,
            mean_path=args.msa_mean_path,
            std_path=args.msa_std_path,
            window_size=args.window_size,
            unit_length=2 ** args.down_t,
            text_embed_dim=args.text_embed_dim,
        )

    if humanml_loader_factory is None:
        from humanml3d_272.dataset_eval_t2m import DATALoader

        humanml_loader_factory = DATALoader
    return humanml_loader_factory(
        args.dataname,
        True,
        32,
        unit_length=2 ** args.down_t,
    )


def _build_network(args):
    return msa_vae.MSA_HumanVAE(
        hidden_size=args.hidden_size,
        down_t=args.down_t,
        stride_t=args.stride_t,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation="relu",
        latent_dim=args.latent_dim,
        clip_range=[-30, 20],
        trans_d_model=args.trans_d_model,
        trans_nhead=args.trans_nhead,
        trans_enc_layers=args.trans_enc_layers,
        trans_dec_layers=args.trans_dec_layers,
        trans_ff_size=args.trans_ff_size,
        trans_dropout=args.trans_dropout,
        clip_dim=args.text_embed_dim,
        disable_decoupling=args.disable_decoupling,
    )


def _load_checkpoint_for_mode(path, args):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "net" not in checkpoint:
        if args.msa_data_mode == "babel_sparse_global":
            raise ValueError("BABEL evaluation requires a checkpoint with MSA metadata")
        return checkpoint

    metadata = checkpoint.get("metadata")
    if metadata is None:
        if args.msa_data_mode == "babel_sparse_global":
            raise ValueError("BABEL evaluation refuses a legacy HumanML checkpoint")
    else:
        validate_msa_checkpoint_metadata(metadata, args)
    return checkpoint["net"]


def _load_humanml_evaluator(device):
    evaluator_root = Path(__file__).resolve().parent / "Evaluator_272"
    sys.path.insert(0, str(evaluator_root))
    from mld.models.architectures.temos.motionencoder.actor import (
        ActorAgnosticEncoder,
    )
    from mld.models.architectures.temos.textencoder.distillbert_actor import (
        DistilbertActorAgnosticEncoder,
    )

    model_path = evaluator_root / "deps" / "distilbert-base-uncased"
    text_encoder = DistilbertActorAgnosticEncoder(
        str(model_path), num_layers=4, latent_dim=256
    )
    motion_encoder = ActorAgnosticEncoder(
        nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300
    )
    evaluator_checkpoint = torch.load(
        evaluator_root
        / "experiments"
        / "temos"
        / "EXP1"
        / "checkpoints"
        / "epoch=99.ckpt",
        map_location="cpu",
    )
    for prefix, encoder in (
        ("textencoder", text_encoder),
        ("motionencoder", motion_encoder),
    ):
        state = {
            key.replace(prefix + ".", ""): value
            for key, value in evaluator_checkpoint["state_dict"].items()
            if key.startswith(prefix + ".")
        }
        encoder.load_state_dict(state, strict=True)
        encoder.eval().to(device)
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    return [text_encoder, motion_encoder]


def main():
    args = option_msa_vae.get_args_parser()
    if args.msa_data_mode == "humanml_full":
        # Preserve the legacy HumanML CLI contract: its launcher passes paths
        # relative to Evaluator_272 and creates the dataset/module symlinks
        # there.  BABEL mode deliberately stays at the repository root.
        os.chdir(Path(__file__).resolve().parent / "Evaluator_272")
    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.out_dir, exist_ok=True)
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    default_text_dim = 512 if args.text_encoder_type == "clip" else 768
    if args.text_embed_dim <= 0:
        args.text_embed_dim = default_text_dim
    args.trans_d_model = args.text_embed_dim
    args.clip_dim = args.text_embed_dim

    state = _load_checkpoint_for_mode(args.resume_pth, args)
    network = _build_network(args)
    network.load_state_dict(state, strict=True)

    if args.msa_data_mode == "babel_sparse_global":
        accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(even_batches=False)
        )
    else:
        accelerator = Accelerator()
    device = accelerator.device
    network.to(device)
    validation_loader = build_validation_loader(args)

    if args.msa_data_mode == "babel_sparse_global":
        validation_dataset = validation_loader.dataset
        validation_loader = prepare_babel_validation_loader(
            accelerator, validation_loader
        )
        network = accelerator.prepare(network)
        evaluate_msa_vae_babel(
            args.out_dir,
            validation_loader,
            accelerator.unwrap_model(network),
            validation_dataset,
            logger,
            writer,
            iteration=0,
            phase=args.phase,
            best_semantic=float("inf"),
            best_mpjpe=float("inf"),
            device=device,
            accelerator=accelerator,
            metadata={
                "msa_data_mode": args.msa_data_mode,
                "local_align_weight": args.local_align_weight,
            },
            save_checkpoints=False,
        )
        writer.close()
        return

    network.eval()
    evaluator = _load_humanml_evaluator(device)
    compatible_network = EvalCompat(network)
    metrics = [[] for _ in range(7)]
    for _ in range(3):
        result = eval_trans.evaluation_msa_vae_single(
            args.out_dir,
            validation_loader,
            compatible_network,
            logger,
            writer,
            evaluator=evaluator,
            device=device,
        )
        for values, metric in zip(metrics, result[:7]):
            values.append(float(metric))

    labels = ("FID", "MPJPE", "Diversity", "R@1", "R@2", "R@3", "MM-dist")
    logger.info("Final HumanML result (mean over 3 repeats):")
    for label, values in zip(labels, metrics):
        logger.info(
            "{}: {:.4f} +/- {:.4f}".format(
                label, float(np.mean(values)), float(np.std(values))
            )
        )
    writer.close()


if __name__ == "__main__":
    main()
