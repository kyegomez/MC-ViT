[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# MC -VIT
Implementation of the model: "(MC-ViT)" from the paper: "Memory Consolidation Enables Long-Context Video Understanding"


## Install
`$ pip install mcvit`

## Usage
```python
import torch
from mcvit.model import MCViT


# Initialize the MCViT model
mcvit = MCViT(
    dim=512,
    attn_seq_len=256,
    dim_head=64,
    dropout=0.1,
    chunks=16,
    depth=12,
    cross_attn_heads=8,
)

# Create a random tensor to represent a video
x = torch.randn(
    1, 3, 256, 256, 256
)  # (batch, channels, frames, height, width)

# Pass the tensor through the model
output = mcvit(x)

print(
    output.shape
)  # Outputs the shape of the tensor after passing through the model

```


# License
MIT
