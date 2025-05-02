"""piowave network model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import numpy as np


class AVRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        pos_encoding_sigma = cfg["pos_encoding_sigma"]
        dir_encoding_sig = cfg["dir_encoding_sig"]
        tx_encoding_sig = cfg["tx_encoding_sig"]

        sigma_encoder_network = cfg["sigma_encoder_network"]
        sigma_decoder_network = cfg["sigma_decoder_network"]
        signal_network = cfg['signal_network']

        self.signal_output_dim = cfg['signal_output_dim']

        self._pos_encoding = tcnn.Encoding(
            3, pos_encoding_sigma, dtype=torch.float32)
        self._dir_encoding = tcnn.Encoding(
            3, dir_encoding_sig, dtype=torch.float32)
        self._tx_encoding = tcnn.Encoding(
            3, tx_encoding_sig, dtype=torch.float32)

        network_in_dims = self._pos_encoding.n_output_dims
        self._model_encoder_sigma = tcnn.Network(
            n_input_dims=network_in_dims,
            n_output_dims=128,
            network_config=sigma_encoder_network,
        )

        self._model_decoder_sigma = tcnn.Network(
            n_input_dims=self._model_encoder_sigma.n_output_dims,
            n_output_dims=1,
            network_config=sigma_decoder_network,
        )

        n_signal_input = self._model_encoder_sigma.n_output_dims + \
            self._dir_encoding.n_output_dims + self._tx_encoding.n_output_dims
        self._model_signal = tcnn.Network(
            n_input_dims=n_signal_input,
            n_output_dims=self.signal_output_dim,
            network_config=signal_network,
        )

    def forward(self, pts, view, tx):
        """forward function of the model

        Parameters
        ----------
        pts: [batchsize, n_rays * n_samples, 3], position of voxels
        view: [batchsize, n_rays * n_samples, 3], view direction
        tx: [batchsize, n_rays * n_samples, 3], position of transmitter, implemented as orientation of the rx data

        Returns
        ----------
        attn: [batchsize, n_rays * n_samples, 1].
        signal: [batchsize, n_rays * n_samples, ir length].
        """
        bs = pts.size(0)
        n_ray_points = pts.size(1)

        pts = (pts.view(-1, 3) + 1)/2
        view = (view.view(-1, 3) + 1)/2
        tx = (tx.view(-1, 3) + 1)/2

        pos_embedding = self._pos_encoding(pts)
        sigma_feature = self._model_encoder_sigma(pos_embedding)
        attn = self._model_decoder_sigma(F.relu(sigma_feature))

        view_embedding = self._dir_encoding(view)
        tx_embedding = self._tx_encoding(tx)

        feature_all = torch.cat(
            [F.relu(sigma_feature), view_embedding, tx_embedding], -1)
        signal = self._model_signal(feature_all)

        attn = abs(F.leaky_relu(attn)).view(bs, n_ray_points, 1)
        signal = signal.reshape(bs, n_ray_points,  self.signal_output_dim)
        return attn, signal


class AVRModel_complex(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.leaky_relu = cfg["leaky_relu"]
        pos_encoding_sigma = cfg["pos_encoding_sigma"]
        tx_pos_encoding_sigma = cfg["tx_pos_encoding_sigma"]

        pos_encoding_signal = cfg['pos_encoding_sig']
        tx_pos_encoding_signal = cfg['tx_pos_encoding_sig']

        dir_encoding_sig = cfg["dir_encoding_sig"]
        tx_dir_encoding_sig = cfg["tx_dir_encoding_sig"]

        sigma_encoder_network = cfg["sigma_encoder_network"]
        sigma_decoder_network = cfg["sigma_decoder_network"]
        signal_network = cfg['signal_network']

        self.signal_output_dim = cfg['signal_output_dim']

        self._pos_encoding = tcnn.Encoding(
            3, pos_encoding_sigma, dtype=torch.float32)
        self._pos_signal_encoding = tcnn.Encoding(
            3, pos_encoding_signal, dtype=torch.float32)
        self._tx_pos_encoding = tcnn.Encoding(
            3, tx_pos_encoding_sigma, dtype=torch.float32)
        self._tx_pos_signal_encoding = tcnn.Encoding(
            3, tx_pos_encoding_signal, dtype=torch.float32)

        self._dir_encoding = tcnn.Encoding(
            3, dir_encoding_sig, dtype=torch.float32)
        self._tx_dir_encoding = tcnn.Encoding(
            3, tx_dir_encoding_sig, dtype=torch.float32)

        network_in_dims = self._pos_encoding.n_output_dims + \
            self._tx_pos_encoding.n_output_dims
        self._model_encoder_sigma = tcnn.Network(
            n_input_dims=network_in_dims,
            n_output_dims=256,
            network_config=sigma_encoder_network,
        )

        self._model_decoder_sigma = tcnn.Network(
            n_input_dims=self._model_encoder_sigma.n_output_dims,
            n_output_dims=1,
            network_config=sigma_decoder_network,
        )

        n_signal_input = self._model_encoder_sigma.n_output_dims + \
            self._dir_encoding.n_output_dims + \
            self._tx_dir_encoding.n_output_dims + \
            self._pos_signal_encoding.n_output_dims + \
            self._tx_pos_signal_encoding.n_output_dims

        self._model_signal = tcnn.Network(
            n_input_dims=n_signal_input,
            n_output_dims=self.signal_output_dim,
            network_config=signal_network,
        )

    def forward(self, pts, view, tx, tx_view):
        """forward function of the model

        Parameters
        ----------
        pts: [batchsize, n_rays * n_samples, 3], position of voxels
        view: [batchsize, n_rays * n_samples, 3], view direction
        tx: [batchsize, n_rays * n_samples, 3], position of emitter
        tx_view: [batchsize, n_rays * n_samples, 3], emitter view direction

        Returns
        ----------
        attn: [batchsize, n_rays * n_samples, 1].
        signal: [batchsize, n_rays * n_samples, ir length].
        """

        bs = pts.size(0)
        n_ray_points = pts.size(1)

        pts = (pts.view(-1, 3) + 1)/2
        view = (view.view(-1, 3) + 1)/2
        tx = (tx.view(-1, 3) + 1)/2
        tx_view = (tx_view.reshape(-1, 3) + 1)/2

        pos_embedding = self._pos_encoding(pts)
        tx_pos_embedding = self._tx_pos_encoding(tx)

        sigma_feature = self._model_encoder_sigma(
            torch.cat([pos_embedding, tx_pos_embedding], -1))
        attn = self._model_decoder_sigma(F.relu(sigma_feature))

        view_embedding = self._dir_encoding(view)
        tx_view_embedding = self._tx_dir_encoding(tx_view)
        signal_embedding = self._pos_signal_encoding(pts)
        tx_signal_embedding = self._tx_pos_signal_encoding(tx)

        feature_all = torch.cat([F.relu(sigma_feature), view_embedding,
                                tx_view_embedding, signal_embedding, tx_signal_embedding], -1)
        signal = self._model_signal(feature_all)

        attn = abs(F.leaky_relu(attn, negative_slope=self.leaky_relu)).view(
            bs, n_ray_points, 1)
        signal = (signal).reshape(bs, n_ray_points,  self.signal_output_dim)
        return attn, signal
