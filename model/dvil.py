import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnetv2_bottleneck import ResnetV2Bottleneck
from .convlstm import ConvLSTM


class HPF(nn.Module): # High-pass filter
    def __init__(self):
        super().__init__()
        # self.weights = torch.nn.Parameter(torch.tensor(
        #     [
        #         [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        #         [[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]],
        #         [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        #     ]
        # ).unsqueeze(1), requires_grad=True)
        self.hpf = nn.Conv2d(3, 9, 3, 1, 1, bias=True)
        torch.nn.init.xavier_normal_(self.hpf.weight)
        torch.nn.init.zeros_(self.hpf.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # y = []
        # for c in range(C):
        #     y.append(F.conv2d(x[:, c, :, :].unsqueeze(1), self.weights, padding=1))
        # y = torch.cat(y, dim=1)
        # return y
        return self.hpf(x)
    

class FIRE(nn.Module): # Flow-guided Inter-frame Residual Extraction
    def __init__(self):
        super().__init__()
        self.requires_grad_(False)

    def forward(self, x, flow):
        T, C, H, W = x.shape
        N = T - 2

        x_prime = torch.zeros((N, C * 2, H, W), dtype=torch.float32, device=x.device, requires_grad=False)

        for t in range(1, N + 1):
            idx_x = torch.arange(0, H).unsqueeze(1).repeat_interleave(W, dim=1).to(x.device)
            idx_y = torch.arange(0, W).repeat(H, 1).to(x.device)
            fwd_flow = flow[t - 1]
            bwd_flow = flow[-t]
            fwd_flow_x, fwd_flow_y = fwd_flow[0, :, :], fwd_flow[1, :, :]
            bwd_flow_x, bwd_flow_y = bwd_flow[0, :, :], bwd_flow[1, :, :]
            idx_x_prime_fwd = torch.clamp(torch.round(idx_x + fwd_flow_x), 0, H - 1).int()
            idx_y_prime_fwd = torch.clamp(torch.round(idx_y + fwd_flow_y), 0, W - 1).int()
            idx_x_prime_bwd = torch.clamp(torch.round(idx_x + bwd_flow_x), 0, H - 1).int()
            idx_y_prime_bwd = torch.clamp(torch.round(idx_y + bwd_flow_y), 0, W - 1).int()
            x_prime_fwd = x[t + 1, :, idx_x_prime_fwd, idx_y_prime_fwd]
            x_prime_bwd = x[t - 1, :, idx_x_prime_bwd, idx_y_prime_bwd]

            x_prime[t - 1] = torch.cat([x[t] - x_prime_fwd, x[t] - x_prime_bwd], dim=0)

        return x_prime
    

class IntraFrameResidual(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.hpf = HPF()
        self.residual_blocks = nn.Sequential(
            ResnetV2Bottleneck(in_chs=9, out_chs=64, bottle_ratio=0.25, stride=2),
            ResnetV2Bottleneck(in_chs=64, out_chs=128, bottle_ratio=0.25, stride=2),
        )

    def forward(self, x):
        x = x[1: -1]
        x = self.hpf(x)
        x = self.residual_blocks(x)
        return x
    

class InterFrameResidual(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.fire = FIRE()
        self.residual_blocks = nn.Sequential(
            ResnetV2Bottleneck(in_chs=6, out_chs=64, bottle_ratio=0.25, stride=2),
            ResnetV2Bottleneck(in_chs=64, out_chs=128, bottle_ratio=0.25, stride=2),
        )

    def forward(self, x, flow):
        x = self.fire(x, flow)
        x = self.residual_blocks(x)
        return x


class DVIL_PyTorch(nn.Module):
    def __init__(
        self,
        img_size,
    ):
        super().__init__()
        self.img_size = img_size

        self.intra_frame_residual = IntraFrameResidual()
        self.inter_frame_residual = InterFrameResidual()

        self.pre_decoder = nn.Sequential(
            ResnetV2Bottleneck(in_chs=256, out_chs=512, bottle_ratio=0.25, stride=2),
            ResnetV2Bottleneck(in_chs=512, out_chs=1024, bottle_ratio=0.25, stride=2),
        )

        self.decoder_convlstm_1 = ConvLSTM(
            img_size=(img_size[0] // 16, img_size[1] // 16),
            input_dim=1024,
            hidden_dim=512,
            kernel_size=(3, 3),
            cnn_dropout=0.0,
            rnn_dropout=0.0,
            peephole=True,
            layer_norm=True,
            bidirectional=True,
            return_sequence=True,
        )
        # self.decoder_conv2dtrp_1 = nn.ConvTranspose2d(1024, 64, 8, 4, 2)
        self.decoder_conv2dtrp_1 = nn.Sequential(
            nn.Conv2d(1024, 64, 3, 1, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )
        self.decoder_convlstm_2 = ConvLSTM(
            img_size=(img_size[0] // 4, img_size[1] // 4),
            input_dim=64,
            hidden_dim=32,
            kernel_size=(3, 3),
            cnn_dropout=0.0,
            rnn_dropout=0.0,
            peephole=True,
            layer_norm=True,
            bidirectional=True,
            return_sequence=True,
        )
        # self.decoder_conv2dtrp_2 = nn.ConvTranspose2d(64, 4, 8, 4, 2)
        self.decoder_conv2dtrp_2 = nn.Sequential(
            nn.Conv2d(64, 4, 3, 1, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

        self.localizer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, flow):
        intra_x = self.intra_frame_residual(x)
        inter_x = self.inter_frame_residual(x, flow)
        x = torch.cat([intra_x, inter_x], dim=1)
        x = self.pre_decoder(x)
        x, _, _ = self.decoder_convlstm_1(x.unsqueeze(0))
        x = x.squeeze(1)
        x = self.decoder_conv2dtrp_1(x)
        x, _, _ = self.decoder_convlstm_2(x.unsqueeze(0))
        x = x.squeeze(1)
        x = self.decoder_conv2dtrp_2(x)
        x = self.localizer(x).squeeze(1)
        return x
        