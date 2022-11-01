# Copyright (C) 2022 Balázs Róbert Glávits

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# long with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import time
from datetime import datetime, timezone
import random

import numpy as np
import torch
import torch_geometric.nn as pyg_nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData

from model import HeteroConvEncoder, InnerProductDecoder, Model, \
    UniformNegativeEdgeSampler, load_and_preprocess_data, train, evaluate, Prediction


class HeteroConvModel(Model):
    def __init__(self):
        super().__init__()
        self._encoder = HeteroConvEncoder(
            dim=[64, 32, 32],
            node_types=['protein', 'disease'],
            conv_factory=self.conv_factory,
            use_pre_mlp=False,
            use_inner_mlp=False,
            use_batch_norm=True,
            dropout=0.1,
            act='PReLU',
            use_post_mlp=False,
            normalize=False
        )
        self._decoder = InnerProductDecoder()

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @staticmethod
    def conv_factory(dim: int):
        return pyg_nn.HeteroConv({
            ('protein', 'associated_with', 'protein'): pyg_nn.GCNConv(-1, dim),
            ('protein', 'associated_with', 'disease'): pyg_nn.SAGEConv((-1, -1), dim, root_weight=True),
            ('disease', 'associated_with', 'protein'): pyg_nn.SAGEConv((-1, -1), dim, root_weight=True),
            ('disease', 'associated_with', 'disease'): pyg_nn.GATConv(-1, dim, heads=1)
        }, aggr='sum')


def main():
    timestamp_unix = int(time.time())
    seed_everything(timestamp_unix)
    timestamp = datetime.fromtimestamp(timestamp_unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    log = SummaryWriter(log_dir=os.path.join('logs', timestamp))

    data_split: dict[str, dict[str, HeteroData]] = load_and_preprocess_data(
        path='protein_disease_hetero_graph.pt',
        protein_principal_components=128, disease_principal_components=128,
        ppi_prune_threshold=0.0, pda_high_quality_threshold=0.1
    )

    # TODO: move tensors to the GPU

    neg_edge_ratio = 1.0
    neg_samplers = {
        k: UniformNegativeEdgeSampler.create_for_data(
            message=data_split[k]['message'], supervision=data_split[k]['supervision'], neg_ratio=neg_edge_ratio)
        for k in ('train', 'val', 'test', 'test_medium', 'test_low')
    }

    model = HeteroConvModel()

    with torch.no_grad():
        x_dict = data_split['train']['message'].x_dict
        message_edge_index_dict = data_split['train']['message'].edge_index_dict
        message_edge_attr_dict = data_split['train']['message'].edge_attr_dict
        # Due to lazy initialization, we need to run one model step so the number of parameters can be inferred.
        model.encode(x_dict, message_edge_index_dict, message_edge_attr_dict)

    print(f'Model contains {sum([np.prod(x.shape, dtype=int) for x in model.parameters()])} parameters.')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    bad_epochs = 0
    for epoch in range(1, 2000):
        train_loss = train(model, optimizer, message=data_split['train']['message'],
                           supervision=data_split['train']['supervision'], sampler=neg_samplers['train'],
                           log=log, log_global_step=epoch)
        val_loss, val_metrics = evaluate(model, message=data_split['val']['message'],
                                         supervision=data_split['val']['supervision'], sampler=neg_samplers['val'],
                                         log=log, log_global_step=epoch)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f} '
              f'{"Improved" if val_loss < best_val_loss else f"No improvement since {bad_epochs + 1} epochs"}.')
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join('checkpoints', f'model_{timestamp}.pt'))
            best_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > 7:
                break
    best_model_state = torch.load(os.path.join('checkpoints', f'model_{timestamp}.pt'))
    model.load_state_dict(best_model_state)

    test_losses = dict()
    for k in ('test', 'test_medium', 'test_low'):
        test_loss, test_metrics = evaluate(model, message=data_split[k]['message'],
                                           supervision=data_split[k]['supervision'], sampler=neg_samplers[k],
                                           log=log, log_group=k)
        test_losses[k] = test_loss
        print(f"Test loss for kind '{k}': {test_loss:.4f}")


@torch.no_grad()
def best_prediction(timestamp):
    random.seed(int(datetime.fromisoformat(timestamp).timestamp()))
    data_split = load_and_preprocess_data(
        path='protein_disease_hetero_graph.pt',
        protein_principal_components=128, disease_principal_components=128,
        ppi_prune_threshold=0.0, pda_high_quality_threshold=0.1)
    random.seed()

    best_model_state = torch.load(os.path.join('checkpoints', f'model_{timestamp}.pt'))
    model = HeteroConvModel()
    model.eval()
    model.load_state_dict(best_model_state)

    pred = model.predict(data_split['test']['message'])

    training_set = data_split['test']['message']['protein', 'disease'].edge_index.numpy()

    return pred, training_set


if __name__ == '__main__':
    main()
