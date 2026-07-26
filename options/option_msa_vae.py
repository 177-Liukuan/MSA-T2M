import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='MSA-VAE (Multi-Scale Semantic Alignment VAE) training',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ## dataloader
    parser.add_argument('--dataname', type=str, default='t2m_272', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')
    parser.add_argument('--use_ft_split', action='store_true', default=True,
                        help='use train_ft.txt (HumanML3D∩BABEL intersection)')
    parser.add_argument('--no_ft_split', dest='use_ft_split', action='store_false',
                        help='use train.txt (full HumanML3D) instead of train_ft.txt')
    parser.add_argument(
        '--msa_data_mode',
        choices=['humanml_full', 'babel_sparse_global'],
        default='humanml_full',
        help='MSA-VAE supervision/data contract',
    )
    parser.add_argument('--bridge_split_file', default='./humanml3d_272/split/train_ft.txt')
    parser.add_argument('--bridge_motion_dir', default='./humanml3d_272/motion_data')
    parser.add_argument('--bridge_text_dir', default='./humanml3d_272/texts')
    parser.add_argument('--bridge_global_embed_dir', default='./humanml3d_272/text_latents_t5')
    parser.add_argument('--bridge_local_embed_dir', default='./humanml3d_272/t5_enc_single')
    parser.add_argument('--babel_train_motion_dir', default='./babel_272_stream/train_stream')
    parser.add_argument('--babel_train_text_dir', default='./babel_272_stream/train_stream_text')
    parser.add_argument('--babel_train_t5_cache_dir', default='./babel_272_stream/t5_enc_single/train')
    parser.add_argument('--babel_train_cache_manifest', default='./babel_272_stream/t5_enc_single/train/manifest.json')
    parser.add_argument('--babel_val_motion_dir', default='./babel_272_stream/val_stream')
    parser.add_argument('--babel_val_text_dir', default='./babel_272_stream/val_stream_text')
    parser.add_argument('--babel_val_t5_cache_dir', default='./babel_272_stream/t5_enc_single/val')
    parser.add_argument('--babel_val_cache_manifest', default='./babel_272_stream/t5_enc_single/val/manifest.json')
    parser.add_argument('--msa_mean_path', default='',
                        help='normalization Mean.npy; empty resolves by msa_data_mode')
    parser.add_argument('--msa_std_path', default='',
                        help='normalization Std.npy; empty resolves by msa_data_mode')

    # variable-length sequence training
    parser.add_argument(
        '--sequence_mode',
        type=str,
        default='window',
        choices=['window', 'full', 'mixed'],
        help='motion view: legacy window, complete sequence, or full+replay',
    )
    parser.add_argument(
        '--full-seq-batch-size', '--full_seq_batch_size',
        dest='full_seq_batch_size',
        default=32,
        type=int,
        help='per-process batch size for complete motion sequences',
    )
    parser.add_argument(
        '--window-replay-interval', '--window_replay_interval',
        dest='window_replay_interval',
        default=4,
        type=int,
        help='use one 64-frame replay batch every N mixed-mode steps',
    )
    parser.add_argument(
        '--length-bucket-size', '--length_bucket_size',
        dest='length_bucket_size',
        default=256,
        type=int,
        help='number of sorted samples per shuffled full-sequence bucket',
    )

    # text encoder selection
    parser.add_argument('--text_encoder_type', type=str, default='t5', choices=['clip', 't5'],
                        help='text encoder type for semantic alignment')
    parser.add_argument('--text_embed_dim', type=int, default=0,
                        help='text embedding dim; <=0 means auto (clip=512, t5=768)')

    # frame-level local text embeddings (offline)
    parser.add_argument('--clip_embed_dir', type=str, default='./humanml3d_272/clip_enc_single',
                        help='local CLIP embedding directory (.npy, T x 512)')
    parser.add_argument('--t5_embed_dir', type=str, default='./humanml3d_272/t5_enc_single',
                        help='local T5 embedding directory (.npy, T x 768)')

    # global text embeddings (offline)
    parser.add_argument('--use_offline_global_text', action='store_true', default=True,
                        help='use precomputed global text embeddings instead of online encoding')
    parser.add_argument('--no_offline_global_text', dest='use_offline_global_text', action='store_false',
                        help='disable offline global text embeddings and use online text encoder')
    parser.add_argument('--clip_global_embed_dir', type=str, default='./humanml3d_272/text_latents_clip',
                        help='global CLIP text embedding directory (.npy per sample)')
    parser.add_argument('--t5_global_embed_dir', type=str, default='./humanml3d_272/text_latents_t5',
                        help='global T5 text embedding directory (.npy per sample)')

    ## optimization
    parser.add_argument('--total-iter', default=2000000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=5e-5, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs='+', type=int,
                        help='learning rate schedule (iterations)')
    parser.add_argument('--gamma', default=0.05, type=float, help='learning rate decay')
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')

    # --- Causal CNN VAE architecture ---
    parser.add_argument('--down-t', type=int, default=2, help='downsampling rate')
    parser.add_argument('--stride-t', type=int, default=2, help='stride size')
    parser.add_argument('--depth', type=int, default=3, help='depth of the CNN network')
    parser.add_argument('--dilation-growth-rate', type=int, default=3, help='dilation growth rate')
    parser.add_argument('--latent_dim', default=16, type=int, help='CNN latent dimension')
    parser.add_argument('--hidden_size', default=1024, type=int, help='CNN hidden size')

    # --- Transformer AE architecture ---
    parser.add_argument('--trans_d_model', default=768, type=int,
                        help='Transformer model dimension (will be forced to text_embed_dim)')
    parser.add_argument('--trans_nhead', default=8, type=int, help='number of attention heads')
    parser.add_argument('--trans_enc_layers', default=4, type=int, help='number of Transformer encoder layers')
    parser.add_argument('--trans_dec_layers', default=4, type=int, help='number of Transformer decoder layers')
    parser.add_argument('--trans_ff_size', default=1024, type=int, help='Transformer feed-forward dimension')
    parser.add_argument('--trans_dropout', default=0.1, type=float, help='Transformer dropout')
    parser.add_argument('--disable_decoupling', action='store_true', default=False,
                        help='ablation: disable dual-track decoupling, use sampled z as Transformer AE input/target')

    # --- text alignment (legacy names kept for compatibility) ---
    parser.add_argument('--clip_dim', default=768, type=int,
                        help='[legacy] alignment feature dimension; auto-overridden by text_embed_dim')
    parser.add_argument('--clip_version', default='ViT-B/32', type=str,
                        help='CLIP model version (for clip mode)')
    parser.add_argument('--t5_model_path', default='sentencet5-xxl/', type=str,
                        help='SentenceT5 model path (for online t5 mode)')
    parser.add_argument('--t5_batch_size', default=32, type=int,
                        help='SentenceT5 online encode batch size')

    # --- Spotlight global alignment ---
    parser.add_argument('--spotlight_alpha', default=-1.0, type=float,
                        help='Spotlight interpolation alpha. '
                             '-1 = dynamic (window_size / total_frames), '
                             '>=0 = fixed value in [0,1]')

    # --- Loss weights ---
    parser.add_argument('--root_loss', default=7.0, type=float, help='root joint loss weight')
    parser.add_argument('--latent_recon_weight', default=1.0, type=float,
                        help='Transformer latent recon loss weight')
    parser.add_argument('--global_align_weight', default=0.5, type=float,
                        help='global text alignment loss weight')
    parser.add_argument('--local_align_weight', default=0.2, type=float,
                        help='local text alignment loss weight')

    ## phased training
    parser.add_argument('--phase', default=0, type=int, choices=[0, 1, 2],
                        help='training phase: 0=all-at-once (legacy), '
                             '1=freeze CNN (only train Transformer+proj), '
                             '2=unfreeze all with differential LR')
    parser.add_argument('--cnn_lr_scale', default=0.1, type=float,
                        help='LR scale factor for CNN params in phase 2')

    ## resume
    parser.add_argument('--resume-pth', type=str, default=None, help='resume pth for MSA-VAE')
    parser.add_argument('--resume-cnn-pth', type=str, default=None,
                        help='resume pth for pretrained Causal CNN VAE (load CNN weights only)')
    parser.add_argument(
        '--resume-cnn-sha256',
        type=str,
        default=None,
        help='externally approved SHA-256 for the joint-domain Causal TAE artifact',
    )

    ## output directory
    parser.add_argument('--out-dir', type=str, default='output/', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='vis', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp', help='name of the experiment')
    parser.add_argument('--latent_dir', type=str, default='t2m_latents/', help='latent directory')

    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=20000, type=int, help='evaluation frequency')
    parser.add_argument(
        '--validation-seed',
        default=123,
        type=int,
        help='fixed RNG seed for deterministic internal validation',
    )
    parser.add_argument(
        '--validation-batch-size',
        default=32,
        type=int,
        help='batch size for deterministic complete-motion validation',
    )
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    parser.add_argument('--nb_joints', default=22, type=int, help='number of joints')
    parser.add_argument('--num_gpus', default=1, type=int, help='number of GPUs')

    args = parser.parse_args()
    if args.validation_batch_size < 1:
        parser.error('--validation-batch-size must be positive')
    if args.msa_data_mode == 'babel_sparse_global':
        if args.phase == 0:
            parser.error('babel_sparse_global supports only --phase 1 or --phase 2')
        if args.text_encoder_type != 't5':
            parser.error('babel_sparse_global requires --text_encoder_type t5')
        if args.text_embed_dim not in (0, 768):
            parser.error('babel_sparse_global requires --text_embed_dim 768 (or 0 to auto-resolve)')
        args.text_embed_dim = 768
        default_mean_path = './babel_272/t2m_babel_mean_std/Mean.npy'
        default_std_path = './babel_272/t2m_babel_mean_std/Std.npy'
    else:
        default_mean_path = './humanml3d_272/mean_std/Mean.npy'
        default_std_path = './humanml3d_272/mean_std/Std.npy'
    if not args.msa_mean_path:
        args.msa_mean_path = default_mean_path
    if not args.msa_std_path:
        args.msa_std_path = default_std_path
    return args
