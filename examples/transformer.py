import torch

import os
import sys
current = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current))

from model.transformer import Transformer
from visual_methods.heatmap import Heatmap
from visual_methods.heatmaps import Heatmaps
from visualization import Visualization

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    examples = [
        # torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0, 0, 3]], dtype=torch.long, device=device),
        # torch.tensor([[2, 1, 1, 1, 1, 1, 1, 1, 1, 3]], dtype=torch.long, device=device),
        # torch.tensor([[2, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
        # torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device)
        # torch.tensor([[2, 0, 1, 3]], dtype=torch.long, device=device)
    ]

    model = Transformer(
        num_tokens = 4,
        dim_model = 8,
        num_heads = 2,
        num_encoder_layers = 3,
        num_decoder_layers = 3,
        dropout_p = 0.1
    )
    model.eval()

    model_input = []
    SOS_token = 2

    for example in examples:
        y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
            
        info = {
            'file' : "simple_transformer",
            'data' : (example, y_input, tgt_mask)
        }

        model_input.append(info)

    visual = Heatmaps()
    visualization = Visualization(model, model_input, visual)
    visualization.visualize()


if __name__ == "__main__":
    main()
