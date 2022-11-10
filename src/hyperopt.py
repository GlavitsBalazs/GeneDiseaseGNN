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
from datetime import datetime

import numpy as np
import optuna.storages
import torch
import torch_geometric.nn as pyg_nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData

from model import Model, HeteroConvEncoder, MLPDecoder, InnerProductDecoder, load_and_preprocess_data, \
    WeightedNegativeEdgeSampler, evaluate, train, UniformNegativeEdgeSampler


class HyperModel(Model):
    def __init__(self, trial: optuna.Trial):
        super().__init__()

        n_layers = trial.suggest_int('n_layers', 1, 3)
        dim = [trial.suggest_int('dim_0', 16, 128, 16)]
        for i in range(1, n_layers):
            dim.append(trial.suggest_int(f'dim_{i}', 16, 128, 16))

        use_pre_mlp = trial.suggest_categorical("use_pre_mlp", [True, False])
        use_inner_mlp = trial.suggest_categorical("use_inner_mlp", [True, False])
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        use_dropout = trial.suggest_categorical("use_dropout", [True, False])
        act = trial.suggest_categorical("use_prelu_act", ['PReLU', 'ReLU'])
        use_post_mlp = trial.suggest_categorical("use_post_mlp", [True, False])

        normalize = trial.suggest_categorical("normalize", [True, False])
        dropout = trial.suggest_float("dropout", 0.2, 0.5) if use_dropout else 0.0

        self._encoder = HeteroConvEncoder(
            dim=dim,
            node_types=['protein', 'disease'],
            conv_factory=self.conv_factory,
            use_pre_mlp=use_pre_mlp,
            use_inner_mlp=use_inner_mlp,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            act=act,
            use_post_mlp=use_post_mlp,
            normalize=normalize
        )
        use_lin_decoder = trial.suggest_categorical("use_lin_decoder", [True, False])
        if use_lin_decoder:
            h = trial.suggest_categorical("mlp_decoder_dim", [16, 32, 64])
            self._decoder = MLPDecoder(in_channels=dim[-1], hidden_channels=h)
        else:
            self._decoder = InnerProductDecoder()

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @staticmethod
    def conv_factory(dim: int):
        # TODO: variable convolutions, aggregations

        return pyg_nn.HeteroConv({
            ('protein', 'associated_with', 'protein'): pyg_nn.GCNConv(-1, dim),
            ('protein', 'associated_with', 'disease'): pyg_nn.SAGEConv((-1, -1), dim, root_weight=True),
            ('disease', 'associated_with', 'protein'): pyg_nn.SAGEConv((-1, -1), dim, root_weight=True),
            ('disease', 'associated_with', 'disease'): pyg_nn.GATConv(-1, dim, heads=1),
        }, aggr='sum')


def objective(trial: optuna.Trial):
    timestamp_unix = int(time.time())
    seed_everything(timestamp_unix)
    timestamp = datetime.fromtimestamp(timestamp_unix).strftime("%Y-%m-%dT%H:%M:%S")

    print(f'Trial {trial.number}: {timestamp}')
    trial.set_user_attr('timestamp', timestamp)
    trial.set_user_attr('seed', timestamp_unix)
    log = SummaryWriter(log_dir=os.path.join('logs', f'{timestamp}'))

    # a new split is created for every trial
    data_split: dict[str, dict[str, HeteroData]] = load_and_preprocess_data(
        path='protein_disease_hetero_graph.pt',
        protein_principal_components=trial.suggest_categorical('protein_principal_components', [16, 32, 64, 128, 256]),
        disease_principal_components=trial.suggest_categorical('disease_principal_components', [16, 32, 64, 128, 256]),
        ppi_prune_threshold=trial.suggest_categorical('ppi_prune_threshold', [0.0, 0.4, 0.7]),
        pda_high_quality_threshold=trial.suggest_categorical('pda_high_quality_threshold', [0.1, 0.2, 0.3])
    )

    # TODO: move all the data to GPU

    neg_edge_ratio = trial.suggest_categorical('neg_edge_ratio', [1.0, 0.8, 1.2, 2.0, 3.0])
    neg_samplers = {
        k: UniformNegativeEdgeSampler.create_for_data(  # todo: hparam for sampler type
            message=data_split[k]['message'], supervision=data_split[k]['supervision'], neg_ratio=neg_edge_ratio)
        for k in ('train', 'val', 'test', 'test_medium', 'test_low')
    }

    model: Model = HyperModel(trial)

    with torch.no_grad():
        x_dict = data_split['train']['message'].x_dict
        message_edge_index_dict = data_split['train']['message'].edge_index_dict
        message_edge_attr_dict = data_split['train']['message'].edge_attr_dict
        # Due to lazy initialization, we need to run one model step so the number of parameters can be inferred.
        model.encode(x_dict, message_edge_index_dict, message_edge_attr_dict)

    print(f'Model contains {sum([np.prod(x.shape, dtype=int) for x in model.parameters()])} parameters.')

    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    # lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    adam_lr = trial.suggest_float("adam_lr", 1e-4, 1e-2, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

    # do not suggest any more hyperparameters beyond this point
    torch.save(trial.params | trial.user_attrs, os.path.join('checkpoints', f'hparams_{timestamp}.pt'))

    best_val_loss = float('inf')
    bad_epochs = 0
    for epoch in range(1, 70):
        try:
            train_loss = train(model, optimizer, message=data_split['train']['message'],
                               supervision=data_split['train']['supervision'], sampler=neg_samplers['train'],
                               log=log, log_global_step=epoch)
            val_loss, val_metrics = evaluate(model, message=data_split['val']['message'],
                                             supervision=data_split['val']['supervision'], sampler=neg_samplers['val'],
                                             log=log, log_global_step=epoch)
            print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f} '
                  f'{"Improved" if val_loss < best_val_loss else f"No improvement since {bad_epochs + 1} epochs"}.')
            trial.report(val_metrics['avep'], epoch)
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), os.path.join('checkpoints', f'model_{timestamp}.pt'))
                best_val_loss = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs > 4:
                    break
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        except KeyboardInterrupt as ki:  # when you don't like how the trial is going just send a ^C
            raise optuna.exceptions.TrialPruned() from ki

    best_model_state = torch.load(os.path.join('checkpoints', f'model_{timestamp}.pt'))
    model.load_state_dict(best_model_state)

    test_losses = dict()
    test_aveps = dict()
    for k in ('test', 'test_medium', 'test_low'):
        test_loss, test_metrics = evaluate(model, message=data_split[k]['message'],
                                           supervision=data_split[k]['supervision'], sampler=neg_samplers[k],
                                           log=log, log_group=k)
        test_losses[k] = test_loss
        test_aveps[k] = test_metrics['avep']
        print(f"Test loss for kind '{k}': {test_loss:.4f}")
    return test_aveps['test']


def main():
    storage = optuna.storages.RDBStorage("sqlite:///optuna.db")
    study = optuna.create_study(
        storage=storage, study_name="GDA-HGNN", direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    optuna.visualization.plot_intermediate_values(study).show()


if __name__ == '__main__':
    main()
