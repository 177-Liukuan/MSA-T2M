import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='TAE-GAN-v1: Decoder-only adversarial fine-tuning of Causal TAE',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument('--dataname',    type=str, default='t2m_272')
    parser.add_argument('--batch-size',  default=64,  type=int)
    parser.add_argument('--window-size', type=int, default=64)

    # ── Optimisation ─────────────────────────────────────────────────────────
    parser.add_argument('--total-iter',    default=200000, type=int)
    parser.add_argument('--warm-up-iter',  default=1000,   type=int)
    parser.add_argument('--lr',            default=1e-5,   type=float,
                        help='decoder learning rate (lower than TAE due to fine-tune)')
    parser.add_argument('--lr-disc',       default=1e-4,   type=float,
                        help='discriminator learning rate')
    parser.add_argument('--lr-scheduler',  default=[150000], nargs='+', type=int)
    parser.add_argument('--gamma',         default=0.1,   type=float)
    parser.add_argument('--weight-decay',  default=0.0,   type=float)

    # ── TAE architecture (must match the checkpoint being loaded) ─────────────
    parser.add_argument('--down-t',              type=int,   default=2)
    parser.add_argument('--stride-t',            type=int,   default=2)
    parser.add_argument('--depth',               type=int,   default=3)
    parser.add_argument('--dilation-growth-rate',type=int,   default=3)
    parser.add_argument('--latent_dim',          type=int,   default=16)
    parser.add_argument('--hidden_size',         type=int,   default=1024)

    # ── GAN hyper-params ──────────────────────────────────────────────────────
    parser.add_argument('--disc-start',   default=10000, type=int,
                        help='global step at which GAN loss is switched on')
    parser.add_argument('--disc-weight',  default=0.5,   type=float,
                        help='adaptive weight scale for adversarial loss')
    parser.add_argument('--fm-weight',    default=10.0,  type=float,
                        help='feature matching loss weight')
    parser.add_argument('--disc-ndf',     default=64,    type=int,
                        help='discriminator base feature channels')
    parser.add_argument('--disc-n-layers',default=3,     type=int)
    parser.add_argument('--disc-freq',     default=1,     type=int,
                        help='update discriminator every N generator steps (default=1)')
    parser.add_argument('--disc-clip-grad', default=1.0,  type=float,
                        help='max grad norm for discriminator (0=disable)')

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    parser.add_argument('--tae-ckpt',    type=str, required=True,
                        help='path to pretrained TAE checkpoint (net_best_mpjpe.pth)')
    parser.add_argument('--resume-pth',  type=str, default=None,
                        help='resume from a TAE-GAN-v1 checkpoint')

    # ── Reconstruction loss ───────────────────────────────────────────────────
    parser.add_argument('--root_loss',   default=7.0, type=float)

    # ── Misc ─────────────────────────────────────────────────────────────────
    parser.add_argument('--out-dir',     type=str, default='output/')
    parser.add_argument('--exp-name',    type=str, default='tae_gan_v1')
    parser.add_argument('--print-iter',  default=200,   type=int)
    parser.add_argument('--eval-iter',   default=10000, type=int)
    parser.add_argument('--seed',        default=123,   type=int)
    parser.add_argument('--num_gpus',    default=1,     type=int)
    parser.add_argument('--nb_joints',   default=22,    type=int)

    return parser
