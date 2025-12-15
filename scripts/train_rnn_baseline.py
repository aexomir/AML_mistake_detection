#!/usr/bin/env python3
"""
Training script for RNN/LSTM baseline (V_RNN) for CaptainCook4D SupervisedER mistake detection.
This script trains an RNN baseline and compares it against existing V1 (MLP) and V2 (Transformer) baselines.
"""
import os
import csv
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim, nn
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torcheval.metrics.functional import binary_auprc
from tqdm import tqdm

from base import fetch_model, fetch_model_name, store_model, save_results, collate_stats, convert_and_round
from core.config import Config
from core.utils import init_logger_and_wandb
from constants import Constants as const
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset, collate_fn_rnn


def train_epoch_rnn(model, device, train_loader, optimizer, epoch, criterion):
    """Train one epoch for RNN model."""
    model.train()
    train_loader = tqdm(train_loader)
    num_batches = len(train_loader)
    train_losses = []

    for batch_idx, batch_data in enumerate(train_loader):
        if len(batch_data) == 3:
            # RNN collate function returns (padded_features, labels, lengths)
            data, target, lengths = batch_data
            data, target, lengths = data.to(device), target.to(device), lengths.to(device)
        else:
            # Fallback for standard collate
            data, target = batch_data
            data, target = data.to(device), target.to(device)
            lengths = None

        assert not torch.isnan(data).any(), "Data contains NaN values"

        optimizer.zero_grad()
        output = model(data, lengths=lengths)
        loss = criterion(output, target)

        assert not torch.isnan(loss).any(), "Loss contains NaN values"

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
        train_loader.set_description(
            f'Train Epoch: {epoch}, Progress: {batch_idx}/{num_batches}, Loss: {loss.item():.6f}'
        )

    return train_losses


def test_er_model_rnn(model, test_loader, criterion, device, phase, step_normalization=True, 
                       sub_step_normalization=True, threshold=0.6):
    """Evaluate RNN model with same metrics as existing baselines."""
    total_samples = 0
    all_targets = []
    all_outputs = []

    test_loader = tqdm(test_loader)
    num_batches = len(test_loader)
    test_losses = []

    test_step_start_end_list = []
    counter = 0

    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                # RNN collate function returns (padded_features, labels, lengths)
                data, target, lengths = batch_data
                data, target, lengths = data.to(device), target.to(device), lengths.to(device)
            else:
                # Fallback for standard collate
                data, target = batch_data
                data, target = data.to(device), target.to(device)
                lengths = None

            output = model(data, lengths=lengths)
            batch_size = data.shape[0]
            total_samples += batch_size
            loss = criterion(output, target)
            test_losses.append(loss.item())

            sigmoid_output = output.sigmoid()
            all_outputs.append(sigmoid_output.detach().cpu().numpy().reshape(-1))
            all_targets.append(target.detach().cpu().numpy().reshape(-1))

            test_step_start_end_list.append((counter, counter + batch_size))
            counter += batch_size

            test_loader.set_description(f'{phase} Progress: {total_samples}/{num_batches}')

    # Flatten lists
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    assert not np.isnan(all_outputs).any(), "Outputs contain NaN values"

    # ------------------------- Sub-Step Level Metrics -------------------------
    all_sub_step_targets = all_targets.copy()
    all_sub_step_outputs = all_outputs.copy()

    # Calculate metrics at the sub-step level
    pred_sub_step_labels = (all_sub_step_outputs > 0.5).astype(int)
    sub_step_precision = precision_score(all_sub_step_targets, pred_sub_step_labels)
    sub_step_recall = recall_score(all_sub_step_targets, pred_sub_step_labels)
    sub_step_f1 = f1_score(all_sub_step_targets, pred_sub_step_labels)
    sub_step_accuracy = accuracy_score(all_sub_step_targets, pred_sub_step_labels)
    sub_step_auc = roc_auc_score(all_sub_step_targets, all_sub_step_outputs)
    sub_step_pr_auc = binary_auprc(torch.tensor(all_sub_step_outputs), torch.tensor(all_sub_step_targets))

    sub_step_metrics = {
        const.PRECISION: sub_step_precision,
        const.RECALL: sub_step_recall,
        const.F1: sub_step_f1,
        const.ACCURACY: sub_step_accuracy,
        const.AUC: sub_step_auc,
        const.PR_AUC: sub_step_pr_auc
    }

    # -------------------------- Step Level Metrics --------------------------
    all_step_targets = []
    all_step_outputs = []

    for start, end in test_step_start_end_list:
        step_output = all_outputs[start:end]
        step_target = all_targets[start:end]

        step_output = np.array(step_output)
        if end - start > 1:
            if sub_step_normalization:
                prob_range = np.max(step_output) - np.min(step_output)
                if prob_range > 0:
                    step_output = (step_output - np.min(step_output)) / prob_range

        mean_step_output = np.mean(step_output)
        step_target = 1 if np.mean(step_target) > 0.95 else 0

        all_step_outputs.append(mean_step_output)
        all_step_targets.append(step_target)

    all_step_outputs = np.array(all_step_outputs)

    if step_normalization:
        prob_range = np.max(all_step_outputs) - np.min(all_step_outputs)
        if prob_range > 0:
            all_step_outputs = (all_step_outputs - np.min(all_step_outputs)) / prob_range

    all_step_targets = np.array(all_step_targets)

    # Calculate metrics at the step level
    pred_step_labels = (all_step_outputs > threshold).astype(int)
    precision = precision_score(all_step_targets, pred_step_labels, zero_division=0)
    recall = recall_score(all_step_targets, pred_step_labels)
    f1 = f1_score(all_step_targets, pred_step_labels)
    accuracy = accuracy_score(all_step_targets, pred_step_labels)

    auc = roc_auc_score(all_step_targets, all_step_outputs)
    pr_auc = binary_auprc(torch.tensor(all_step_outputs), torch.tensor(all_step_targets))

    step_metrics = {
        const.PRECISION: precision,
        const.RECALL: recall,
        const.F1: f1,
        const.ACCURACY: accuracy,
        const.AUC: auc,
        const.PR_AUC: pr_auc
    }

    # Print step level metrics
    print("----------------------------------------------------------------")
    print(f'{phase} Sub Step Level Metrics: {sub_step_metrics}')
    print(f"{phase} Step Level Metrics: {step_metrics}")
    print("----------------------------------------------------------------")

    return test_losses, sub_step_metrics, step_metrics


def train_rnn_baseline(config):
    """Main training function for RNN baseline."""
    torch.manual_seed(config.seed)

    # Setup data loaders with RNN collate function
    cuda_kwargs = {
        "num_workers": 8,
        "pin_memory": False,
    }
    train_kwargs = {**cuda_kwargs, "shuffle": True, "batch_size": config.batch_size}
    test_kwargs = {**cuda_kwargs, "shuffle": False, "batch_size": 1}

    print("-------------------------------------------------------------")
    print("Training RNN baseline model and testing on step level")
    print(f"Train args: {train_kwargs}")
    print(f"Test args: {test_kwargs}")
    print(f"RNN Config: hidden_size={getattr(config, 'rnn_hidden_size', 256)}, "
          f"num_layers={getattr(config, 'rnn_num_layers', 2)}, "
          f"bidirectional={getattr(config, 'rnn_bidirectional', True)}, "
          f"rnn_type={getattr(config, 'rnn_type', 'LSTM')}")
    print(config.args)
    print("-------------------------------------------------------------")

    train_dataset = CaptainCookStepDataset(config, const.TRAIN, config.split)
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn_rnn, **train_kwargs)
    val_dataset = CaptainCookStepDataset(config, const.VAL, config.split)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn_rnn, **test_kwargs)
    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn_rnn, **test_kwargs)

    # Initialize model
    model = fetch_model(config)
    device = config.device
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5], dtype=torch.float32).to(device))
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        factor=0.1, patience=5, verbose=True,
        threshold=1e-4, threshold_mode="abs", min_lr=1e-7
    )

    # Initialize variables to track the best model
    best_model = {'model_state': None, 'metric': 0}

    model_name = config.model_name
    if config.model_name is None:
        model_name = fetch_model_name(config)
        config.model_name = model_name

    train_stats_directory = f"stats/{config.task_name}/{config.variant}/{config.backbone}"
    os.makedirs(train_stats_directory, exist_ok=True)
    train_stats_file = f"{model_name}_training_performance.txt"
    train_stats_file_path = os.path.join(train_stats_directory, train_stats_file)

    # Training loop
    with open(train_stats_file_path, 'w') as f:
        f.write('Epoch, Train Loss, Val Loss, Test Loss, Precision, Recall, F1, AUC\n')
        for epoch in range(1, config.num_epochs + 1):
            # Train
            train_losses = train_epoch_rnn(model, device, train_loader, optimizer, epoch, criterion)

            # Validate
            val_losses, sub_step_metrics, step_metrics = test_er_model_rnn(
                model, val_loader, criterion, device, phase='val'
            )

            scheduler.step(step_metrics[const.AUC])

            # Test
            test_losses, test_sub_step_metrics, test_step_metrics = test_er_model_rnn(
                model, test_loader, criterion, device, phase='test'
            )

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_test_loss = sum(test_losses) / len(test_losses)

            precision = step_metrics['precision']
            recall = step_metrics['recall']
            f1 = step_metrics['f1']
            auc = step_metrics['auc']

            # Write losses and metrics to file
            f.write(
                f'{epoch}, {avg_train_loss:.6f}, {avg_val_loss:.6f}, {avg_test_loss:.6f}, '
                f'{precision:.6f}, {recall:.6f}, {f1:.6f}, {auc:.6f}\n'
            )

            running_metrics = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "val_loss": avg_val_loss,
                "val_metrics": {
                    "step_metrics": step_metrics,
                    "sub_step_metrics": sub_step_metrics
                },
                "test_metrics": {
                    "step_metrics": test_step_metrics,
                    "sub_step_metrics": test_sub_step_metrics
                }
            }

            if config.enable_wandb:
                wandb.log(running_metrics)

            print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, '
                  f'Test Loss: {avg_test_loss:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, '
                  f'F1: {f1:.6f}, AUC: {auc:.6f}')

            # Update best model based on AUC
            if auc > best_model['metric']:
                best_model['metric'] = auc
                best_model['model_state'] = model.state_dict()

            store_model(model, config, ckpt_name=f"{model_name}_epoch_{epoch}.pt")

        # Save the best model
        if best_model['model_state'] is not None:
            model.load_state_dict(best_model['model_state'])
            store_model(model, config, ckpt_name=f"{model_name}_best.pt")

    # Final evaluation and save results
    print("\n" + "="*60)
    print("Final Evaluation Results")
    print("="*60)
    final_test_losses, final_sub_step_metrics, final_step_metrics = test_er_model_rnn(
        model, test_loader, criterion, device, phase='test', threshold=0.6
    )
    
    # Save results to CSV (same format as existing baselines)
    save_results(config, final_sub_step_metrics, final_step_metrics, 
                 step_normalization=True, sub_step_normalization=True, threshold=0.6)

    # Print comparison note
    print("\n" + "="*60)
    print("Comparison Note:")
    print("="*60)
    print("V_RNN (RNN/LSTM) results are saved to results/error_recognition/combined_results/")
    print("Compare against V1 (MLP) and V2 (Transformer) using the same CSV file.")
    print("Ensure you use the same split, backbone, and threshold for fair comparison.")
    print("="*60)


def main():
    """Main entry point."""
    conf = Config()
    conf.task_name = const.ERROR_RECOGNITION
    conf.variant = const.RNN_VARIANT  # Set to RNN variant
    
    # Set RNN hyperparameters (can be overridden via CLI)
    if not hasattr(conf, 'rnn_hidden_size'):
        conf.rnn_hidden_size = 256
    if not hasattr(conf, 'rnn_num_layers'):
        conf.rnn_num_layers = 2
    if not hasattr(conf, 'rnn_dropout'):
        conf.rnn_dropout = 0.2
    if not hasattr(conf, 'rnn_bidirectional'):
        conf.rnn_bidirectional = True
    if not hasattr(conf, 'rnn_use_attention'):
        conf.rnn_use_attention = False
    if not hasattr(conf, 'rnn_type'):
        conf.rnn_type = 'LSTM'
    
    if conf.model_name is None:
        m_name = fetch_model_name(conf)
        conf.model_name = m_name

    if conf.enable_wandb:
        init_logger_and_wandb(conf)

    train_rnn_baseline(conf)

    if conf.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

