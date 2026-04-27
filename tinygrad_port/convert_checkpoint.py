#!/usr/bin/env python3
"""Convert PyTorch checkpoint to tinygrad safetensor format."""
import argparse
import torch
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import safe_save


def convert_pytorch_to_tinygrad(pytorch_path, tinygrad_path):
    """Load PyTorch state_dict, convert to tinygrad safetensors."""
    pt_ckpt = torch.load(pytorch_path, map_location='cpu')

    tg_state = {}

    # Convert each component
    for component in ['compressor', 'atoms', 'hypernet', 'confidence_head',
                       'answer_head', 'message_generator', 'ordinal_head']:
        if component in pt_ckpt:
            for name, param in pt_ckpt[component].items():
                key = f"{component}.{name}"
                tg_state[key] = Tensor(param.numpy())

    safe_save(tg_state, tinygrad_path)
    print(f"Converted {len(tg_state)} parameters to {tinygrad_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='PyTorch .pt checkpoint')
    parser.add_argument('--output', required=True, help='Output .safetensor path')
    args = parser.parse_args()
    convert_pytorch_to_tinygrad(args.input, args.output)
