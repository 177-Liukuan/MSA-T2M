import json
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from eval_msa_t2m_rag_t5 import (
    OfflineTextEmbeddingLookup,
    RAGRetriever,
    load_state_strip_module,
    parse_args,
    resolve_data_root,
    resolve_existing_path,
    resolve_reference_end_latent,
)
from humanml3d_272 import dataset_eval_t2m
from models import msa_vae
from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model import LLaMARAGWrapper
from utils import utils_model
from utils.eval_msa_t2m_optimized import (
    OptimizedRAGEvalSampler,
    evaluation_transformer_272_optimized,
)


DEFAULT_OPTIMIZED_EXPERIMENT = "MotionStreamer_t2m_272_msa_rag_t5_optimized"


def build_optimized_parser(argv=None):
    arguments = list(argv) if argv is not None else sys.argv[1:]
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + arguments
        args = parse_args()
    finally:
        sys.argv = original_argv

    has_explicit_name = any(
        argument == "--exp-name" or argument.startswith("--exp-name=")
        for argument in arguments
    )
    if not has_explicit_name:
        args.exp_name = DEFAULT_OPTIMIZED_EXPERIMENT

    if args.generative_head_type != "ddpm":
        raise ValueError(
            "The protocol-equivalent optimized pipeline supports the official "
            "DDPM evaluation head only."
        )

    args.optimized_pipeline = True
    args.generation_batch_size = 32
    return args


def _resolve_runtime_paths(args):
    args.resume_pth = (
        resolve_existing_path(args.resume_pth)
        if args.resume_pth is not None
        else None
    )
    args.resume_trans = (
        resolve_existing_path(args.resume_trans)
        if args.resume_trans is not None
        else None
    )
    args.latent_dir = resolve_existing_path(args.latent_dir, must_be_dir=True)
    args.text_latent_dir = resolve_existing_path(
        args.text_latent_dir,
        must_be_dir=True,
    )
    args.hcls_dir = resolve_existing_path(args.hcls_dir, must_be_dir=True)
    args.empty_text_path = resolve_existing_path(args.empty_text_path)
    args.t5_model_path = resolve_existing_path(
        args.t5_model_path,
        must_be_dir=True,
    )
    return args


def _build_msa_vae(args, device):
    net = msa_vae.MSA_HumanVAE(
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
        clip_dim=args.clip_dim,
    )

    print("loading MSA-VAE checkpoint from {}".format(args.resume_pth))
    checkpoint = torch.load(args.resume_pth, map_location="cpu")
    state = (
        checkpoint["net"]
        if isinstance(checkpoint, dict) and "net" in checkpoint
        else checkpoint
    )
    net.load_state_dict(state, strict=True)
    net.eval()
    return net.to(device)


def _build_rag_model(args, device, logger):
    config = LLaMAHFConfig.from_name("Normal_size")
    config.block_size = 78
    base_model = LLaMAHF(
        config,
        args.num_diffusion_head_layers,
        args.latent_dim,
        device,
    )
    rag_model = LLaMARAGWrapper(
        base_model=base_model,
        model_dim=config.n_embd,
        disable_rag=args.disable_rag,
    )

    if args.resume_trans is None:
        raise ValueError("Please provide --resume-trans for RAG checkpoint.")
    print("loading RAG checkpoint from {}".format(args.resume_trans))
    checkpoint = torch.load(args.resume_trans, map_location="cpu")

    trans_key = (
        "trans_ema"
        if args.use_ema and "trans_ema" in checkpoint
        else "trans"
    )
    rag_key = "rag_ema" if args.use_ema and "rag_ema" in checkpoint else "rag"

    if trans_key not in checkpoint:
        raise KeyError("RAG checkpoint missing key: trans/trans_ema")
    rag_model.base_model.load_state_dict(
        load_state_strip_module(checkpoint[trans_key]),
        strict=False,
    )

    if rag_key in checkpoint:
        rag_model.load_state_dict(
            load_state_strip_module(checkpoint[rag_key]),
            strict=False,
        )
    elif not args.disable_rag:
        raise KeyError("RAG checkpoint missing key: rag/rag_ema")
    else:
        logger.info(
            "Checkpoint has no rag key, continue in no-RAG ablation mode."
        )

    if args.use_ema:
        logger.info(
            "EMA eval enabled. loaded keys: {}, {}".format(
                trans_key,
                rag_key,
            )
        )

    rag_model.eval()
    return rag_model.to(device)


def _load_empty_text_embedding(args, device):
    empty_text_path = args.empty_text_path
    if not os.path.exists(empty_text_path):
        candidates = [
            os.path.join(args.text_latent_dir, "empty_text_embedding.npy"),
            os.path.join(args.text_latent_dir, "empty_cfg_text_t5.npy"),
            os.path.join(args.text_latent_dir, "empty_cfg_text_clip.npy"),
        ]
        empty_text_path = next(
            (path for path in candidates if os.path.exists(path)),
            empty_text_path,
        )

    if not os.path.exists(empty_text_path):
        raise FileNotFoundError(
            "empty cfg text file not found: {}".format(args.empty_text_path)
        )

    embedding = (
        torch.from_numpy(np.load(empty_text_path).astype(np.float32))
        .reshape(-1)
        .to(device)
    )
    if embedding.shape[0] != args.text_embed_dim:
        raise ValueError(
            "empty text embedding dim must be {}, got {} from {}".format(
                args.text_embed_dim,
                embedding.shape[0],
                empty_text_path,
            )
        )
    return embedding


def _build_text_source(args, data_root, logger):
    if args.text_source == "online_t5":
        from sentence_transformers import SentenceTransformer

        text_encoder = SentenceTransformer(args.t5_model_path)
        text_encoder.eval()
        logger.info(
            "Text source: online_t5, model={}".format(args.t5_model_path)
        )
        return None, text_encoder

    text_lookup = OfflineTextEmbeddingLookup(
        data_root=data_root,
        text_latent_dir=args.text_latent_dir,
        split=args.eval_split,
        text_embed_dim=args.text_embed_dim,
    )
    logger.info("Text source: offline precomputed latents")
    return text_lookup, None


def _load_reference_end_latent(args, device, logger):
    reference_path, reference_end = resolve_reference_end_latent(args)
    if not args.enable_stopping:
        logger.info("Continuous stopping disabled by flag.")
        return None
    if reference_end is None:
        raise FileNotFoundError(
            "Cannot locate reference end latent. Provide "
            "--reference_end_latent_path or ensure it exists under "
            "--latent_dir."
        )

    reference = torch.from_numpy(reference_end).float().to(device)
    if reference.shape != (args.latent_dim,):
        raise ValueError(
            "reference end latent dim mismatch: got {}, expected {}".format(
                reference.shape[0],
                args.latent_dim,
            )
        )
    logger.info(
        "Using continuous stopping with reference latent: {}, threshold={}".format(
            reference_path,
            args.stop_threshold,
        )
    )
    return reference


def _load_evaluator(device):
    from mld.models.architectures.temos.motionencoder.actor import (
        ActorAgnosticEncoder,
    )
    from mld.models.architectures.temos.textencoder.distillbert_actor import (
        DistilbertActorAgnosticEncoder,
    )

    textencoder = DistilbertActorAgnosticEncoder(
        "./deps/distilbert-base-uncased",
        num_layers=4,
        latent_dim=256,
    )
    motionencoder = ActorAgnosticEncoder(
        nfeats=272,
        vae=True,
        num_layers=4,
        latent_dim=256,
        max_len=300,
    )

    checkpoint_path = "../Evaluator_272/epoch=99.ckpt"
    print("Loading evaluator checkpoint from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    text_state = {
        key.replace("textencoder.", ""): value
        for key, value in checkpoint["state_dict"].items()
        if key.split(".")[0] == "textencoder"
    }
    motion_state = {
        key.replace("motionencoder.", ""): value
        for key, value in checkpoint["state_dict"].items()
        if key.split(".")[0] == "motionencoder"
    }
    textencoder.load_state_dict(text_state, strict=True)
    motionencoder.load_state_dict(motion_state, strict=True)
    textencoder.eval()
    motionencoder.eval()
    return [textencoder.to(device), motionencoder.to(device)]


def main(argv=None):
    os.chdir("Evaluator_272")
    sys.path.insert(0, os.getcwd())
    device = torch.device("cuda")

    args = build_optimized_parser(argv)
    args = _resolve_runtime_paths(args)
    data_root = resolve_data_root()
    torch.manual_seed(args.seed)

    args.out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.out_dir, exist_ok=True)
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info("optimized_pipeline=true")

    val_loader = dataset_eval_t2m.DATALoader(
        args.dataname,
        args.eval_split == "test",
        32,
    )
    net = _build_msa_vae(args, device)
    rag_model = _build_rag_model(args, device, logger)
    empty_text_emb = _load_empty_text_embedding(args, device)
    text_lookup, text_encoder = _build_text_source(args, data_root, logger)

    retriever = None
    if not args.disable_rag:
        retriever = RAGRetriever(
            args.hcls_dir,
            topk=args.retrieval_topk,
            text_embed_dim=args.text_embed_dim,
            device=device,
        )
    else:
        logger.info("No-RAG ablation enabled: retrieval library bypassed.")

    reference_end_latent = _load_reference_end_latent(args, device, logger)
    trans_for_eval = OptimizedRAGEvalSampler(
        rag_model=rag_model,
        retriever=retriever,
        empty_text_emb=empty_text_emb,
        latent_dim=args.latent_dim,
        device=device,
        reference_end_latent=reference_end_latent,
        stop_threshold=args.stop_threshold,
        enable_stopping=args.enable_stopping,
        text_source=args.text_source,
        text_lookup=text_lookup,
        text_encoder=text_encoder,
        text_embed_dim=args.text_embed_dim,
        disable_rag=args.disable_rag,
        use_random_topk_inference=args.use_random_topk_inference,
    )
    evaluator = _load_evaluator(device)

    fid, diversity, top1, top2, top3, matching, logger = (
        evaluation_transformer_272_optimized(
            val_loader,
            net,
            trans_for_eval,
            logger,
            evaluator,
            cfg=args.cfg_scale,
            device=device,
        )
    )
    logger.info("final result:")
    logger.info("fid: [{}]".format(fid))
    logger.info("div: [{}]".format(diversity))
    logger.info("top1: [{}]".format(top1))
    logger.info("top2: [{}]".format(top2))
    logger.info("top3: [{}]".format(top3))
    logger.info("MM-dist (matching score) : [{}]".format(matching))
    writer.close()


if __name__ == "__main__":
    main()
