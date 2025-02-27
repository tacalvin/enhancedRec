from convlstm import ConvLSTM
from unicodedata import bidirectional
import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, reduce, repeat

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                 embed_onehot.sum(0), alpha=1 - self.decay
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_( embed_sum, alpha= 1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class DQLR(nn.Module):
    def __init__(
        self,
        in_channel=1,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        #not sure why theres two encoder steps
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        #conv to get the correct embedding dim
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.upsample_t2 = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec2 = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.upsample_t3 = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec3 = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        '''self.lstm2 = nn.LSTM(embed_dim + embed_dim,embed_dim + embed_dim)
        self.lstm3 = nn.LSTM(embed_dim + embed_dim,embed_dim + embed_dim)'''

    def forward(self, input):
        print("IN FORWARD INPUT_SIZE:{}".format(input.shape))
        quant_t, quant_b, diff, _, _ = self.encode(input)
        print("QUANT_T:{} QUANT_B:{} DIFF:{}".format(quant_t.shape, quant_b.shape, diff.shape))
        dec  = []
        dec = self.decode(quant_t, quant_b)
        print("DEC SIZE:{}".format(dec.shape))
        dec2 = self.decode2(quant_t, quant_b)
        print("DEC2 SIZE:{}".format(dec2.shape))
        dec3 = self.decode3(quant_t, quant_b)
        print("DEC3 SIZE:{}".format(dec3.shape))
        quit()
        return [dec, dec2, dec3], diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        print("ENC_B:{}".format(enc_b.size()))
        print("ENC_T:{}".format(enc_t.size()))
        # quit()
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode2(self, quant_t, quant_b):
        quant_t = quant_t.detach()
        quant_b = quant_b.detach()
        upsample_t = self.upsample_t2(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec2 = self.dec2(quant)

        return dec2

    def decode3(self, quant_t, quant_b):
        quant_t = quant_t.detach()
        quant_b = quant_b.detach()
        upsample_t = self.upsample_t3(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec3 = self.dec3(quant)

        return dec3
    

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

    '''def LSTM2(self,quant_t,quant_b):
        quant = torch.cat([quant_t, quant_b], 1)
        #quant_lstm = self.lstm2('''
        

class DQLRnn(nn.Module):
    def __init__(
        self,
        in_channel=1,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        #not sure why theres two encoder steps
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        #conv to get the correct embedding dim
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        # self.upsample_t2 = nn.ConvTranspose2d(
        #     embed_dim, embed_dim, 4, stride=2, padding=1
        # )
        # self.dec2 = Decoder(
        #     embed_dim + embed_dim,
        #     in_channel,
        #     channel,
        #     n_res_block,
        #     n_res_channel,
        #     stride=4,
        # )
        # self.upsample_t3 = nn.ConvTranspose2d(
        #     embed_dim, embed_dim, 4, stride=2, padding=1
        # )
        # self.dec3 = Decoder(
        #     embed_dim + embed_dim,
        #     in_channel,
        #     channel,
        #     n_res_block,
        #     n_res_channel,
        #     stride=4,
        # )

        print("Embed",embed_dim)
        # self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        #input_dim = embed_dim by pure coincidence
        self.lstm1 = ConvLSTM(embed_dim * 2, [embed_dim], (3,3), 1, True)
        self.lstm2 = ConvLSTM(embed_dim * 2, [embed_dim], (3,3), 1, True)

    # get a sequence
    def forward(self, input):
        batch_size = input.size()[0]
        seq_length = input.size()[1]
        input = rearrange(input, 'b f c w h -> (b f) c w h')
        quant_t, quant_b, diff, _, _ = self.encode(input, batch_size, seq_length)
        # print(quant_t.size(), quant_b.size())


        
        # quant_t = rearrange(quant_t, '(b f) c w h -> b f c w h', b= batch_size, f=seq_length)
        # print("QUANT:{}", quant_t.size())
        # 
        # 
        # quit()
        # pass through the encoder the encoder should return the enc_b and enc_t but run the lstm enc_t
        out = self.decode(quant_t, quant_b)
        out = rearrange(out, '(b f) c w h -> b f c w h', b= batch_size, f=seq_length)
        # print("OUT:{}".format(out.size()))
        # quit()
        return out, diff


    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def encode(self, input, batch_size, seq_length):
        # print("INPUT_SHAPE:{}".format(input.size()))
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        # print("ENC_B:{}".format(enc_b.size()))
        # print("ENC_T:{}".format(enc_t.size()))
        # quit()
        enc_t = rearrange(enc_t, '(b f) c w h -> b f c w h', b= batch_size, f=seq_length)
        # enc_t = torch.flatten(enc_t, 2)
        # print("ENC_T LSTMIN:{}".format(enc_t.size()))
        out_lstm,_ = self.lstm1(enc_t)
        out_lstmb,_ = self.lstm2(torch.flip(enc_t,(1,)))
        # print("LSTM:",out_lstm[0].size(),out_lstmb[0].size())
        enc_t = torch.cat([out_lstm[0], out_lstmb[0]],2)
        # print("ENC_T:{}".format(enc_t.size()))
        enc_t = rearrange(enc_t, 'b f c w h -> (b f) c w h')

        # quit()
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b