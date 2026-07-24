import sys
import argparse
import warnings
import torch

import eval_msa_t2m_rag_t5 as base_eval

warnings.filterwarnings('ignore')


def _parse_rf_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--generative_head_type', type=str, default=None, choices=['ddpm', 'rectified_flow'])
    parser.add_argument('--num_flow_steps', type=int, default=50)
    parser.add_argument('--flow_solver', type=str, default='euler')
    parser.add_argument('--rf_time_sampling', type=str, default='uniform')
    parser.add_argument('--rf_loss_type', type=str, default='mse')
    return parser.parse_known_args(argv)


def _extract_resume_trans_from_argv(argv):
    """Read --resume-trans value without consuming it from argv."""
    for i, token in enumerate(argv):
        if token.startswith('--resume-trans='):
            return token.split('=', 1)[1]
        if token.startswith('--resume_trans='):
            return token.split('=', 1)[1]
        if token in ('--resume-trans', '--resume_trans'):
            if i + 1 < len(argv):
                return argv[i + 1]
            return None
    return None




def _upsert_cli_arg(argv, key, value):
    """Insert or replace --key VALUE in argv list."""
    out = []
    i = 0
    flag = f'--{key}'
    prefix = f'--{key}='
    while i < len(argv):
        token = argv[i]
        if token == flag:
            i += 2
            continue
        if token.startswith(prefix):
            i += 1
            continue
        out.append(token)
        i += 1
    out.extend([flag, str(value)])
    return out

def _infer_head_from_ckpt(ckpt_path):
    if ckpt_path is None:
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except Exception:
        return None

    if isinstance(ckpt, dict):
        head = ckpt.get('generative_head_type', None)
        if isinstance(head, str):
            return head
    return None


def main():
    rf_args, remaining_argv = _parse_rf_args(sys.argv[1:])
    resume_trans_path = _extract_resume_trans_from_argv(sys.argv[1:])

    inferred_head = _infer_head_from_ckpt(resume_trans_path)
    chosen_head = rf_args.generative_head_type or inferred_head or 'rectified_flow'

    orig_llama_hf = base_eval.LLaMAHF

    class LLaMAHFWithRF(orig_llama_hf):
        def __init__(self, *args, **kwargs):
            # Force RF bridge args to override base defaults from option_trans.
            kwargs['generative_head_type'] = chosen_head
            kwargs['num_flow_steps'] = rf_args.num_flow_steps
            kwargs['flow_solver'] = rf_args.flow_solver
            kwargs['rf_time_sampling'] = rf_args.rf_time_sampling
            kwargs['rf_loss_type'] = rf_args.rf_loss_type
            super().__init__(*args, **kwargs)

    base_eval.LLaMAHF = LLaMAHFWithRF

    orig_get_logger = base_eval.utils_model.get_logger

    def get_logger_with_rf_banner(out_dir):
        logger = orig_get_logger(out_dir)
        logger.info(
            'RF eval bridge config: '
            f'generative_head_type={chosen_head}, '
            f'num_flow_steps={rf_args.num_flow_steps}, '
            f'flow_solver={rf_args.flow_solver}, '
            f'rf_time_sampling={rf_args.rf_time_sampling}, '
            f'rf_loss_type={rf_args.rf_loss_type}'
        )
        return logger

    base_eval.utils_model.get_logger = get_logger_with_rf_banner

    # Keep base_eval.parse_args logs consistent with RF bridge effective config.
    effective_argv = list(remaining_argv)
    effective_argv = _upsert_cli_arg(effective_argv, 'generative_head_type', chosen_head)
    effective_argv = _upsert_cli_arg(effective_argv, 'num_flow_steps', rf_args.num_flow_steps)
    effective_argv = _upsert_cli_arg(effective_argv, 'flow_solver', rf_args.flow_solver)
    effective_argv = _upsert_cli_arg(effective_argv, 'rf_time_sampling', rf_args.rf_time_sampling)
    effective_argv = _upsert_cli_arg(effective_argv, 'rf_loss_type', rf_args.rf_loss_type)

    sys.argv = [sys.argv[0]] + effective_argv
    base_eval.main()


if __name__ == '__main__':
    main()
