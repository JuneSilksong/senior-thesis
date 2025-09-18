import torch.nn as nn
import torch.nn.functional as F
import math
import julius

def rescale_conv(conv, reference):
    """Rescale initial weight scale. It is unclear why it helps but it certainly does.
    """
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            rescale_conv(sub, reference)

class Demucs(nn.Module):
    def __init__(self,
                 sources,
                 # Channels
                 audio_channels=2,
                 channels=64,
                 growth=2,
                 # Main structure
                 depth=6,
                 # Convolutions
                 kernel_size=8,
                 stride=4,
                 context=1,
                 # Normalization
                 norm_groups=4,
                 # Pre/post processing
                 normalize=True,
                 resample=True,
                 # Weight initializaiton
                 rescale = 0.1,
                 ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.channels = channels
        self.growth = growth
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.context = context
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.normalize = normalize
        self.resample = resample

        self.rescale = rescale

        in_channels = audio_channels
        padding = 0

        for index in range(depth):
            # Encoder
            encode = []
            encode += [
                # Conv 1
                nn.Conv1d(in_channels, channels, kernel_size, stride),
                nn.GroupNorm(norm_groups, channels),
                nn.ReLU(),
                # Conv 2
                nn.Conv1d(channels, 2*channels, 1, 1),
                nn.GroupNorm(norm_groups, 2*channels),
                nn.GLU(dim=1),
            ]

            self.encoder.append(nn.Sequential(*encode))

            # Decoder
            decode = []
            if index > 0:
                out_channels = in_channels
                decode += [
                    # Conv 1
                    nn.Conv1d(channels, 2*channels, 2*context+1, 1, padding=context),
                    nn.GroupNorm(norm_groups, 2*channels),
                    nn.GLU(dim=1),
                    # Conv 2
                    nn.ConvTranspose1d(channels, out_channels, kernel_size, stride),
                    nn.GroupNorm(norm_groups, out_channels),
                    nn.ReLU(),
                ]
            else:
                out_channels = len(self.sources) * audio_channels
                decode += [
                    # Conv 1
                    nn.Conv1d(channels, 2*channels, 2*context+1, 1, padding=context),
                    nn.GroupNorm(norm_groups, 2*channels),
                    nn.GLU(dim=1),
                    # Conv 2 (without activation)
                    nn.ConvTranspose1d(channels, out_channels, kernel_size, stride),
                ]

            self.decoder.insert(0, nn.Sequential(*decode))

            in_channels = channels
            channels = growth*channels
        
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=in_channels, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(2 * in_channels, in_channels)

        if rescale:
            rescale_module(self, reference=rescale)
    
    def valid_length(self, length):

        if self.resample:
            length *= 2

        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)

        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        
        if self.resample:
            length = math.ceil(length/2)

        return int(length)

    def forward(self, x):
        # Get valid L_in (T)
        length = x.shape[-1] # (B, C, T) -> T

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            x = (x - mean) / (1e-5 + std)
        else:
            mean = 0
            std = 1
        
        delta = self.valid_length(length) - length
        x = F.pad(x, (delta // 2, delta - delta // 2))

        if self.resample:
            x = julius.resample_frac(x, 1, 2)

        # Encode
        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        # Bidirectional LSTM
        x = x.permute(2, 0, 1) # LSTM takes tensor of shape (T, B, C)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0) # return to (B, C, T)
        
        # Decode
        for decode in self.decoder:
            skip = saved.pop(-1)
            x = decode(x + skip)

        if self.resample:
            x = julius.resample_frac(x, 2, 1)    

        x = x * std + mean
        x = x[..., delta // 2:-(delta - delta // 2)]
        x = x.view(x.size(0), self.audio_channels, x.size(-1)) # FOR DAE
        # x = x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1)) # FOR SEPARATION

        return x