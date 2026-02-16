import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Parameter):
        nn.init.normal_(m, std=0.02)

class MLPPosEmbed(nn.Module):
    def __init__(self, pos_embed: str, activation: str = "gelu"):
        super().__init__()
        self.pos_embed = pos_embed
        self.depth = int(pos_embed.split('_')[-1])
        self.dim = int(pos_embed.split('_')[-2])
        self.input_dim = int(pos_embed.split('_')[-3])
        self.output_dim = self.dim

        # Layers: [in->dim] + (depth-1) times [dim->dim]
        self.mlp = nn.ModuleList(
            [nn.Linear(self.input_dim, self.dim)] +
            [nn.Linear(self.dim, self.dim) for _ in range(self.depth - 1)]
        )

        # Activation (GELU by default)
        if activation.lower() == "gelu":
            self.act = nn.GELU()  # approximate='none' by default
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.mlp:
            init_weights(m)  # your existing initializer

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: (N, input_dim) e.g., (N, 3)
        returns: (N, dim)
        """
        x = points
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            # Apply GELU on all but the final layer
            if i < len(self.mlp) - 1:
                x = self.act(x)
        return x