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

    ## optimization
    parser.add_argument('--total-iter', default=2000000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=5e-5, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')

    # --- Causal CNN VAE architecture ---
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--depth", type=int, default=3, help="depth of the CNN network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument('--latent_dim', default=16, type=int, help='CNN latent dimension')
    parser.add_argument('--hidden_size', default=1024, type=int, help='CNN hidden size')

    # --- Transformer AE architecture ---
    parser.add_argument('--trans_d_model', default=512, type=int, help='Transformer model dimension')
    parser.add_argument('--trans_nhead', default=8, type=int, help='number of attention heads')
    parser.add_argument('--trans_enc_layers', default=4, type=int, help='number of Transformer encoder layers')
    parser.add_argument('--trans_dec_layers', default=4, type=int, help='number of Transformer decoder layers')
    parser.add_argument('--trans_ff_size', default=1024, type=int, help='Transformer feed-forward dimension')
    parser.add_argument('--trans_dropout', default=0.1, type=float, help='Transformer dropout')

    # --- CLIP alignment ---
    parser.add_argument('--clip_dim', default=512, type=int, help='CLIP feature dimension')
    parser.add_argument('--clip_version', default='ViT-B/32', type=str, help='CLIP model version')

    # --- Loss weights ---
    parser.add_argument('--root_loss', default=7.0, type=float, help='root joint loss weight')
    parser.add_argument('--latent_recon_weight', default=1.0, type=float, help='Transformer latent recon loss weight')
    parser.add_argument('--global_align_weight', default=0.5, type=float, help='global CLIP alignment loss weight')
    parser.add_argument('--local_align_weight', default=0.2, type=float, help='local CLIP alignment loss weight')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for MSA-VAE')
    parser.add_argument("--resume-cnn-pth", type=str, default=None, help='resume pth for pretrained Causal CNN VAE (load CNN weights only)')

    ## output directory
    parser.add_argument('--out-dir', type=str, default='output/', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='vis', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp', help='name of the experiment')
    parser.add_argument('--latent_dir', type=str, default='t2m_latents/', help='latent directory')

    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=20000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    parser.add_argument('--nb_joints', default=22, type=int, help='number of joints')
    parser.add_argument('--num_gpus', default=1, type=int, help='number of GPUs')

    return parser.parse_args()
