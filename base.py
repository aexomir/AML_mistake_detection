import csv
import os
from collections import defaultdict

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import wandb
from torch import optim, nn
from torch.utils.data import DataLoader

from constants import Constants as const
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torcheval.metrics.functional import binary_auprc
from tqdm import tqdm

from core.models.blocks import fetch_input_dim, MLP
from core.models.er_former import ErFormer
from dataloader.CaptainCookStepDataset import collate_fn, CaptainCookStepDataset
from dataloader.CaptainCookSubStepDataset import CaptainCookSubStepDataset


def fetch_model_name(config):
    if config.task_name == const.ERROR_CATEGORY_RECOGNITION:
        return fetch_model_name_ecr(config)
    elif config.task_name in  [const.EARLY_ERROR_RECOGNITION, const.ERROR_RECOGNITION]:
        if config.model_name is None:
            if config.backbone in [const.RESNET3D, const.X3D, const.SLOWFAST, const.OMNIVORE]:
                config.model_name = f"{config.task_name}_{config.split}_{config.backbone}_{config.variant}_{config.modality[0]}"
            elif config.backbone == const.IMAGEBIND:
                combined_modality_name = '_'.join(config.modality)
                config.model_name = f"{config.task_name}_{config.split}_{config.backbone}_{config.variant}_{combined_modality_name}"
    return config.model_name


def fetch_model_name_ecr(config):
    combined_modality_name = '_'.join(config.modality)
    if config.model_name is None:
        config.model_name = (f"{config.task_name}_{config.split}_{config.backbone}"
                             f"_{config.variant}_{combined_modality_name}_{config.error_category}")
    return config.model_name


def fetch_model(config):
    model = None
    if config.variant == const.MLP_VARIANT:
        if config.backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, const.SLOWFAST, const.IMAGEBIND]:
            input_dim = fetch_input_dim(config)
            model = MLP(input_dim, 512, 1)
    elif config.variant == const.TRANSFORMER_VARIANT:
        if config.backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, const.SLOWFAST, const.IMAGEBIND]:
            model = ErFormer(config)

    assert model is not None, f"Model not found for variant: {config.variant} and backbone: {config.backbone}"
    model.to(config.device)
    return model


def convert_and_round(value):
    value = value * 100.0
    if isinstance(value, torch.Tensor):
        return np.round(value.numpy(), 2)
    return np.round(value, 2)


def collate_stats(config, sub_step_metrics, step_metrics):
    collated_stats = [config.split, config.backbone, config.variant, config.modality]
    for metric in [const.PRECISION, const.RECALL, const.F1, const.ACCURACY, const.AUC, const.PR_AUC]:
        collated_stats.append(convert_and_round(sub_step_metrics[metric]))
    for metric in [const.PRECISION, const.RECALL, const.F1, const.ACCURACY, const.AUC, const.PR_AUC]:
        # Round to two digits before appending
        collated_stats.append(convert_and_round(step_metrics[metric]))
    return collated_stats


def save_results_to_csv(config, sub_step_metrics, step_metrics, step_normalization=False, sub_step_normalization=False,
                        threshold=0.5):
    results_dir = os.path.join(os.getcwd(), const.RESULTS)
    task_results_dir = os.path.join(results_dir, config.task_name, "combined_results")
    os.makedirs(task_results_dir, exist_ok=True)
    config.model_name = fetch_model_name(config)

    results_file_path = os.path.join(task_results_dir,
                                     f'step_{step_normalization}_substep_{sub_step_normalization}_threshold_{threshold}.csv')
    collated_stats = collate_stats(config, sub_step_metrics, step_metrics)

    file_exist = os.path.isfile(results_file_path)

    with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
        writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
        if not file_exist:
            writer.writerow([
                "Split", "Backbone", "Variant", "Modality",
                "Sub-Step Precision", "Sub-Step Recall", "Sub-Step F1", "Sub-Step Accuracy", "Sub-Step AUC",
                "Sub-Step PR AUC",
                "Step Precision", "Step Recall", "Step F1", "Step Accuracy", "Step AUC", "Step PR AUC"
            ])
        writer.writerow(collated_stats)


def save_error_type_analysis_to_csv(config, error_type_metrics, step_normalization=False, 
                                     sub_step_normalization=False, threshold=0.5):
    """
    Save error type analysis results to CSV file.
    
    Args:
        config: Configuration object
        error_type_metrics: Dictionary mapping error type names to their metrics
        step_normalization: Whether step normalization was used
        sub_step_normalization: Whether sub-step normalization was used
        threshold: Threshold used for binary classification
    """
    if not error_type_metrics:
        return
    
    results_dir = os.path.join(os.getcwd(), const.RESULTS)
    task_results_dir = os.path.join(results_dir, config.task_name, "error_type_analysis")
    os.makedirs(task_results_dir, exist_ok=True)
    config.model_name = fetch_model_name(config)
    
    results_file_path = os.path.join(
        task_results_dir,
        f'error_type_analysis_step_{step_normalization}_substep_{sub_step_normalization}_threshold_{threshold}.csv'
    )
    
    file_exists = os.path.isfile(results_file_path)
    
    with open(results_file_path, "a", newline='') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        
        if not file_exists:
            # Write header
            writer.writerow([
                "Split", "Backbone", "Variant", "Modality",
                "Error Type", "Count",
                "Precision", "Recall", "F1", "Accuracy", "AUC", "PR AUC"
            ])
        
        # Sort error types for consistent output
        sorted_error_types = sorted(error_type_metrics.keys())
        
        for error_type in sorted_error_types:
            metrics = error_type_metrics[error_type]
            row = [
                config.split,
                config.backbone,
                config.variant,
                config.modality[0] if isinstance(config.modality, list) else config.modality,
                error_type,
                metrics.get('count', 0),
                convert_and_round(metrics.get(const.PRECISION, 0.0)),
                convert_and_round(metrics.get(const.RECALL, 0.0)),
                convert_and_round(metrics.get(const.F1, 0.0)),
                convert_and_round(metrics.get(const.ACCURACY, 0.0)),
                convert_and_round(metrics.get(const.AUC, 0.0)),
                convert_and_round(metrics.get(const.PR_AUC, 0.0))
            ]
            writer.writerow(row)


def save_results(config, sub_step_metrics, step_metrics, step_normalization=False, sub_step_normalization=False,
                 threshold=0.5, error_type_metrics=None):
    # 1. Save evaluation results to csv
    save_results_to_csv(config, sub_step_metrics, step_metrics, step_normalization, sub_step_normalization, threshold)
    
    # 2. Save error type analysis results to csv
    if error_type_metrics:
        save_error_type_analysis_to_csv(config, error_type_metrics, step_normalization, 
                                        sub_step_normalization, threshold)


def store_model(model, config, ckpt_name: str):
    task_directory = os.path.join(config.ckpt_directory, config.task_name)
    os.makedirs(task_directory, exist_ok=True)

    variant_directory = os.path.join(task_directory, config.variant)
    os.makedirs(variant_directory, exist_ok=True)

    backbone_directory = os.path.join(variant_directory, config.backbone)
    os.makedirs(backbone_directory, exist_ok=True)

    ckpt_file_path = os.path.join(backbone_directory, ckpt_name)
    torch.save(model.state_dict(), ckpt_file_path)


# ----------------------- TRAIN BASE FILES -----------------------


def train_epoch(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loader = tqdm(train_loader)
    num_batches = len(train_loader)
    train_losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        assert not torch.isnan(data).any(), "Data contains NaN values"

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        assert not torch.isnan(loss).any(), "Loss contains NaN values"

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        train_losses.append(loss.item())
        train_loader.set_description(
            f'Train Epoch: {epoch}, Progress: {batch_idx}/{num_batches}, Loss: {loss.item():.6f}'
        )

    return train_losses


def train_model_base(train_loader, val_loader, config, test_loader=None):
    model = fetch_model(config)
    device = config.device
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5], dtype=torch.float32).to(device))
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        factor=0.1, patience=5, verbose=True,
        threshold=1e-4, threshold_mode="abs", min_lr=1e-7
    )
    # criterion = nn.BCEWithLogitsLoss()
    # Initialize variables to track the best model based on the desired metric (e.g., AUC)
    best_model = {'model_state': None, 'metric': 0}

    model_name = config.model_name
    if config.model_name is None:
        model_name = fetch_model_name(config)
        config.model_name = model_name

    train_stats_directory = f"stats/{config.task_name}/{config.variant}/{config.backbone}"
    os.makedirs(train_stats_directory, exist_ok=True)
    train_stats_file = f"{model_name}_training_performance.txt"
    train_stats_file_path = os.path.join(train_stats_directory, train_stats_file)

    # Open a file to store the losses and metrics
    with open(train_stats_file_path, 'w') as f:
        f.write('Epoch, Train Loss, Test Loss, Precision, Recall, F1, AUC\n')
        for epoch in range(1, config.num_epochs + 1):

            model.train()
            train_loader = tqdm(train_loader)
            num_batches = len(train_loader)
            train_losses = []

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                assert not torch.isnan(data).any(), "Data contains NaN values"

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)

                if torch.isnan(loss).any():
                    print(f"Loss contains NaN values in epoch {epoch}, batch {batch_idx}")
                    continue

                # assert not torch.isnan(loss).any(), "Loss contains NaN values"

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                train_losses.append(loss.item())
                train_loader.set_description(
                    f'Train Epoch: {epoch}, Progress: {batch_idx}/{num_batches}, Loss: {loss.item():.6f}'
                )

            val_losses, sub_step_metrics, step_metrics, val_error_type_metrics = test_er_model(
                model, val_loader, criterion, device, phase='val'
            )

            scheduler.step(step_metrics[const.AUC])

            test_error_type_metrics = None
            test_losses = []
            test_sub_step_metrics = {}
            test_step_metrics = {}
            if test_loader is not None:
                test_losses, test_sub_step_metrics, test_step_metrics, test_error_type_metrics = test_er_model(
                    model, test_loader, criterion, device, phase='test'
                )

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_test_loss = sum(test_losses) / len(test_losses) if test_losses else 0.0

            precision = step_metrics['precision']
            recall = step_metrics['recall']
            f1 = step_metrics['f1']
            auc = step_metrics['auc']

            # Write losses and metrics to file
            f.write(
                f'{epoch}, {avg_train_loss:.6f}, {avg_val_loss:.6f}, {avg_test_loss:.6f}, {precision:.6f}, {recall:.6f}, {f1:.6f}, {auc:.6f}\n')

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

            print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, '
                  f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}, AUC: {auc:.6f}')

            # Update best model based on the chosen metric, here using AUC as an example
            if auc > best_model['metric']:
                best_model['metric'] = auc
                best_model['model_state'] = model.state_dict()

            store_model(model, config, ckpt_name=f"{model_name}_epoch_{epoch}.pt")

        # Save the best model
        if best_model['model_state'] is not None:
            model.load_state_dict(best_model['model_state'])
            store_model(model, config, ckpt_name=f"{model_name}_best.pt")


def train_step_test_step_dataset_base(config):
    torch.manual_seed(config.seed)

    cuda_kwargs = {
        "num_workers": 8,
        "pin_memory": False,
    }
    train_kwargs = {**cuda_kwargs, "shuffle": True, "batch_size": config.batch_size}
    test_kwargs = {**cuda_kwargs, "shuffle": False, "batch_size": 1}

    print("-------------------------------------------------------------")
    print("Training step model and testing on step level")
    print(f"Train args: {train_kwargs}")
    print(f"Test args: {test_kwargs}")
    if config.error_category is not None:
        print(f"Error Category: {config.error_category}")
    print(config.args)
    print("-------------------------------------------------------------")

    train_dataset = CaptainCookStepDataset(config, const.TRAIN, config.split)
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, **train_kwargs)
    val_dataset = CaptainCookStepDataset(config, const.VAL, config.split)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, **test_kwargs)
    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, **test_kwargs)

    return train_loader, val_loader, test_loader


def train_sub_step_test_step_dataset_base(config):
    torch.manual_seed(config.seed)

    cuda_kwargs = {
        "num_workers": 1,
        "pin_memory": False,
    }
    train_kwargs = {**cuda_kwargs, "shuffle": True, "batch_size": 1024}
    test_kwargs = {**cuda_kwargs, "shuffle": False, "batch_size": 1}

    train_dataset = CaptainCookSubStepDataset(config, const.TRAIN, config.split)
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, **train_kwargs)
    val_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, **test_kwargs)
    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, **test_kwargs)

    print("-------------------------------------------------------------")
    print("Training sub-step model and testing on step level")
    print(f"Train args: {train_kwargs}")
    print(f"Test args: {test_kwargs}")
    print(f"Split: {config.split}")
    print("-------------------------------------------------------------")

    return train_loader, val_loader, test_loader


# ----------------------- TEST BASE FILES -----------------------


def compute_error_type_metrics(all_step_targets, all_step_outputs, test_step_error_categories, threshold):
    """
    Compute metrics for each error type.
    
    Args:
        all_step_targets: Array of binary targets (0 or 1) for each step
        all_step_outputs: Array of prediction probabilities for each step
        test_step_error_categories: List of sets, where each set contains error category labels for a step
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary mapping error type names to their metrics
    """
    # Map error labels to names
    error_label_to_name = const.ERROR_LABEL_TO_NAME
    
    # Initialize storage for each error type
    error_type_data = defaultdict(lambda: {'targets': [], 'outputs': []})
    
    # Also track "No Error" category
    no_error_data = {'targets': [], 'outputs': []}
    
    # Group steps by error type
    for step_idx, (target, output, error_categories) in enumerate(
        zip(all_step_targets, all_step_outputs, test_step_error_categories)
    ):
        # error_categories is a set of error labels (e.g., {2, 3} or {0})
        if error_categories is None or len(error_categories) == 0:
            # If no error categories, treat as no error
            no_error_data['targets'].append(target)
            no_error_data['outputs'].append(output)
        else:
            # Filter out label 0 and check if there are actual error labels
            error_labels = {label for label in error_categories if label > 0}
            
            if len(error_labels) == 0:
                # Step has no errors (only label 0 or empty)
                no_error_data['targets'].append(target)
                no_error_data['outputs'].append(output)
            else:
                # Step has errors - add to all relevant error type categories
                # Note: A step can belong to multiple error categories if it has multiple error types
                for error_label in error_labels:
                    error_name = error_label_to_name.get(error_label, f"Unknown_{error_label}")
                    error_type_data[error_name]['targets'].append(target)
                    error_type_data[error_name]['outputs'].append(output)
    
    # Compute metrics for each error type
    error_type_metrics = {}
    
    # Process "No Error" category
    if len(no_error_data['targets']) > 0:
        no_error_targets = np.array(no_error_data['targets'])
        no_error_outputs = np.array(no_error_data['outputs'])
        no_error_preds = (no_error_outputs > threshold).astype(int)
        
        error_type_metrics[const.NO_ERROR] = {
            const.PRECISION: precision_score(no_error_targets, no_error_preds, zero_division=0),
            const.RECALL: recall_score(no_error_targets, no_error_preds, zero_division=0),
            const.F1: f1_score(no_error_targets, no_error_preds, zero_division=0),
            const.ACCURACY: accuracy_score(no_error_targets, no_error_preds),
            const.AUC: roc_auc_score(no_error_targets, no_error_outputs) if len(np.unique(no_error_targets)) > 1 else 0.0,
            const.PR_AUC: float(binary_auprc(torch.tensor(no_error_preds), torch.tensor(no_error_targets)).item()) if len(no_error_targets) > 0 else 0.0,
            'count': len(no_error_targets)
        }
    
    # Process each error type
    for error_name, data in error_type_data.items():
        if len(data['targets']) == 0:
            continue
            
        error_targets = np.array(data['targets'])
        error_outputs = np.array(data['outputs'])
        error_preds = (error_outputs > threshold).astype(int)
        
        # Only compute AUC if we have both classes
        try:
            auc = roc_auc_score(error_targets, error_outputs) if len(np.unique(error_targets)) > 1 else 0.0
        except ValueError:
            auc = 0.0
        
        error_type_metrics[error_name] = {
            const.PRECISION: precision_score(error_targets, error_preds, zero_division=0),
            const.RECALL: recall_score(error_targets, error_preds, zero_division=0),
            const.F1: f1_score(error_targets, error_preds, zero_division=0),
            const.ACCURACY: accuracy_score(error_targets, error_preds),
            const.AUC: auc,
            const.PR_AUC: float(binary_auprc(torch.tensor(error_preds), torch.tensor(error_targets)).item()) if len(error_targets) > 0 else 0.0,
            'count': len(error_targets)
        }
    
    return error_type_metrics


def print_error_type_metrics(error_type_metrics, phase):
    """Print error type metrics in a formatted way."""
    if not error_type_metrics:
        return
    
    print("----------------------------------------------------------------")
    print(f"{phase} Error Type Analysis:")
    print("----------------------------------------------------------------")
    print(f"{'Error Type':<25} {'Count':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'AUC':<12}")
    print("-" * 95)
    
    # Sort by error type name for consistent output
    sorted_error_types = sorted(error_type_metrics.keys())
    
    for error_type in sorted_error_types:
        metrics = error_type_metrics[error_type]
        count = metrics.get('count', 0)
        precision = metrics.get(const.PRECISION, 0.0)
        recall = metrics.get(const.RECALL, 0.0)
        f1 = metrics.get(const.F1, 0.0)
        accuracy = metrics.get(const.ACCURACY, 0.0)
        auc = metrics.get(const.AUC, 0.0)
        
        print(f"{error_type:<25} {count:<8} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} "
              f"{accuracy:<12.4f} {auc:<12.4f}")
    
    print("----------------------------------------------------------------")


def test_er_model(model, test_loader, criterion, device, phase, step_normalization=True, sub_step_normalization=True,
                  threshold=0.6):
    total_samples = 0
    all_targets = []
    all_outputs = []

    test_loader = tqdm(test_loader)
    num_batches = len(test_loader)
    test_losses = []

    test_step_start_end_list = []
    test_step_error_categories = []  # Store error categories for each step
    counter = 0

    with torch.no_grad():
        for batch_data in test_loader:
            # Handle both old format (data, target) and new format (data, target, error_categories)
            if len(batch_data) == 3:
                data, target, error_category_labels = batch_data
            else:
                data, target = batch_data
                error_category_labels = None
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_samples += data.shape[0]
            loss = criterion(output, target)
            test_losses.append(loss.item())

            sigmoid_output = output.sigmoid()
            all_outputs.append(sigmoid_output.detach().cpu().numpy().reshape(-1))
            all_targets.append(target.detach().cpu().numpy().reshape(-1))

            # Store error categories for this batch (one per step in the batch)
            if error_category_labels is not None:
                # error_category_labels is a tuple of sets, one per step in the batch
                test_step_error_categories.extend(error_category_labels)

            test_step_start_end_list.append((counter, counter + data.shape[0]))
            counter += data.shape[0]

            # Set the description of the tqdm instance to show the loss
            test_loader.set_description(f'{phase} Progress: {total_samples}/{num_batches}')

    # Flatten lists
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    # Assert that none of the outputs are NaN
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
    sub_step_pr_auc = binary_auprc(torch.tensor(pred_sub_step_labels), torch.tensor(all_sub_step_targets))

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

    # threshold_outputs = all_outputs / max_probability

    for start, end in test_step_start_end_list:
        step_output = all_outputs[start:end]
        step_target = all_targets[start:end]

        # sorted_step_output = np.sort(step_output)
        # # Top 50% of the predictions
        # threshold = np.percentile(sorted_step_output, 50)
        # step_output = step_output[step_output > threshold]

        # pos_output = step_output[step_output > 0.5]
        # neg_output = step_output[step_output <= 0.5]
        #
        # if len(pos_output) > len(neg_output):
        #     step_output = pos_output
        # else:
        #     step_output = neg_output
        step_output = np.array(step_output)
        # # Scale the output to [0, 1]
        if start - end > 1:
            if sub_step_normalization:
                prob_range = np.max(step_output) - np.min(step_output)
                step_output = (step_output - np.min(step_output)) / prob_range

        mean_step_output = np.mean(step_output)
        step_target = 1 if np.mean(step_target) > 0.95 else 0

        all_step_outputs.append(mean_step_output)
        all_step_targets.append(step_target)

    all_step_outputs = np.array(all_step_outputs)

    # # Scale the output to [0, 1]
    if step_normalization:
        prob_range = np.max(all_step_outputs) - np.min(all_step_outputs)
        all_step_outputs = (all_step_outputs - np.min(all_step_outputs)) / prob_range

    all_step_targets = np.array(all_step_targets)

    # Calculate metrics at the step level
    pred_step_labels = (all_step_outputs > threshold).astype(int)
    precision = precision_score(all_step_targets, pred_step_labels, zero_division=0)
    recall = recall_score(all_step_targets, pred_step_labels)
    f1 = f1_score(all_step_targets, pred_step_labels)
    accuracy = accuracy_score(all_step_targets, pred_step_labels)

    auc = roc_auc_score(all_step_targets, all_step_outputs)
    pr_auc = binary_auprc(torch.tensor(pred_step_labels), torch.tensor(all_step_targets))

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

    # -------------------------- Error Type Analysis --------------------------
    error_type_metrics = None
    if test_step_error_categories and len(test_step_error_categories) == len(all_step_targets):
        error_type_metrics = compute_error_type_metrics(
            all_step_targets, all_step_outputs, test_step_error_categories, threshold
        )
        print_error_type_metrics(error_type_metrics, phase)
    elif test_step_error_categories:
        print(f"Warning: Mismatch between error categories ({len(test_step_error_categories)}) "
              f"and step targets ({len(all_step_targets)}). Skipping error-type analysis.")

    return test_losses, sub_step_metrics, step_metrics, error_type_metrics
