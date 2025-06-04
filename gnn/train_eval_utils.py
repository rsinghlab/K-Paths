import torch
import torch.nn.functional as F
import logging
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import numpy as np
import os

def train_model(model, edge_classifier, optimizer, scheduler,
                combined_train_edge_index, combined_train_edge_type,
                train_edge_index, train_edge_type,
                val_edge_index, val_edge_type,
                num_classes, epochs, log_file, device, patience=50,
                max_samples_per_epoch=1000, num_samples_per_class=10):
    """
    Trains the RGCN and EdgeClassifier models with early stopping and stratified sampling.

    The training procedure includes:
    - Balanced class sampling from training edges.
    - Mixed-precision training with automatic loss scaling.
    - Periodic validation and metric logging.
    - Early stopping based on validation loss.

    Saves the best model (based on validation loss) to 'outputs/best_model.pt'.

    Parameters:
        model (torch.nn.Module): RGCN model for node embedding.
        edge_classifier (torch.nn.Module): Classifier for edge type prediction.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        combined_train_edge_index (torch.Tensor): Edge indices for the full graph (used for embedding).
        combined_train_edge_type (torch.Tensor): Edge types for the full graph.
        train_edge_index (torch.Tensor): Edge indices used for classifier training.
        train_edge_type (torch.Tensor): Edge labels used for classifier training.
        val_edge_index (torch.Tensor): Edge indices used for validation.
        val_edge_type (torch.Tensor): Edge labels used for validation.
        num_classes (int): Number of edge classes.
        epochs (int): Maximum number of training epochs.
        log_file (str): Path to save per-epoch training logs.
        device (torch.device): Device used for computation.
        patience (int): Early stopping patience in epochs.
        max_samples_per_epoch (int): Maximum samples drawn from training set per epoch.
        num_samples_per_class (int): Maximum samples drawn per class per epoch.

    Returns:
        Tuple[List[float]]: Training losses, validation losses, validation accuracies,
                            validation macro-F1 scores, and Cohen’s kappa scores.
    """
    best_loss = float('inf')
    trigger_times = 0
    scaler = GradScaler()

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    val_kappas = []

    # Open log file
    with open(log_file, 'w') as f:
        f.write("Epoch, Loss, Val Loss, Val Acc, Val F1, Val Kappa\n")

    num_training_edges = train_edge_type.size(0)
    logging.info(f"Total training edges available: {num_training_edges}")

    for epoch in range(1, epochs + 1):
        model.train()
        edge_classifier.train()
        optimizer.zero_grad()

        # Stratified sampling to balance classes in training
        sampled_indices = []
        total_samples = 0
        for class_label in range(num_classes):
            class_indices = torch.where(train_edge_type == class_label)[0]
            num_available = class_indices.size(0)
            if num_available > 0:
                num_samples = min(num_samples_per_class, num_available)
                sampled_class_indices = class_indices[torch.randperm(num_available)[:num_samples]]
                sampled_indices.append(sampled_class_indices)
                total_samples += num_samples
        if sampled_indices:
            sampled_indices = torch.cat(sampled_indices)
        else:
            logging.warning("No samples were selected in stratified sampling.")
            continue
        sampled_indices = sampled_indices.to(device)

        # Limit total samples per epoch
        if total_samples > max_samples_per_epoch:
            sampled_indices = sampled_indices[torch.randperm(total_samples)[:max_samples_per_epoch]]

        sampled_edge_index = train_edge_index[:, sampled_indices].to(device)
        sampled_edge_type = train_edge_type[sampled_indices].to(device)

        # Forward pass and loss computation
        with autocast():
            node_embeddings = model(combined_train_edge_index, combined_train_edge_type)
            preds = edge_classifier(node_embeddings, sampled_edge_index)
            loss = F.cross_entropy(preds, sampled_edge_type)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step(loss)

        # Validation step
        model.eval()
        edge_classifier.eval()
        with torch.no_grad():
            node_embeddings = model(combined_train_edge_index, combined_train_edge_type)

            val_preds = edge_classifier(node_embeddings, val_edge_index)
            val_loss = F.cross_entropy(val_preds, val_edge_type)
            _, val_predicted = torch.max(val_preds, dim=1)
            val_predicted = val_predicted.cpu().numpy()
            val_true = val_edge_type.cpu().numpy()

            val_accuracy = accuracy_score(val_true, val_predicted)
            val_f1 = f1_score(val_true, val_predicted, average='macro')
            val_kappa = cohen_kappa_score(val_true, val_predicted)

        # Log metrics
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        val_kappas.append(val_kappa)

        with open(log_file, 'a') as f:
            f.write(f"{epoch}, {loss.item()}, {val_loss.item()}, {val_accuracy}, {val_f1}, {val_kappa}\n")

        # Early stopping based on validation loss
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            trigger_times = 0
            os.makedirs("outputs", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'edge_classifier_state_dict': edge_classifier.state_dict()
            }, 'outputs/best_model.pt')
            logging.info(f"Epoch {epoch}: New best validation loss {best_loss:.4f}. Model saved.")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                logging.info("Early stopping triggered.")
                break

        # Logging progress
        if epoch % 10 == 0 or epoch == 1:
            logging.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Kappa: {val_kappa:.4f}")

        torch.cuda.empty_cache()

        if device.type == 'cuda':
            mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
            mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
            logging.info(f"Epoch {epoch}: GPU Memory Allocated: {mem_alloc:.2f} GB, Reserved: {mem_reserved:.2f} GB")

    # Return metrics for visualization
    return train_losses, val_losses, val_accuracies, val_f1_scores, val_kappas

def evaluate_model(model, edge_classifier, edge_index, edge_type, log_file, results_file, combined_train_edge_index, combined_train_edge_type, device):
    """
    Evaluates a trained RGCN + EdgeClassifier model on a labeled edge set.

    Generates node embeddings using the full training graph, runs the edge
    classifier on the evaluation edge set, and logs accuracy, macro-F1, and Cohen’s kappa.

    Parameters:
        model (torch.nn.Module): Trained RGCN model.
        edge_classifier (torch.nn.Module): Trained edge classifier.
        edge_index (torch.Tensor): Edge indices to evaluate.
        edge_type (torch.Tensor): Ground-truth edge labels.
        log_file (str): Path to write human-readable evaluation results.
        results_file (str): Path to write evaluation metrics (for automated scripts).
        combined_train_edge_index (torch.Tensor): Edge indices for embedding context.
        combined_train_edge_type (torch.Tensor): Edge types for embedding context.
        device (torch.device): Device used for computation.
    """
    model.eval()
    edge_classifier.eval()
    with torch.no_grad():
        node_embeddings = model(combined_train_edge_index.to(device), combined_train_edge_type.to(device))
        preds = edge_classifier(node_embeddings, edge_index.to(device))
        
        # Derive num_classes from the EdgeClassifier's final layer
        num_classes = edge_classifier.fc2.out_features  # Retrieve number of classes from fc2 layer
        
        # Validate target labels
        min_label = edge_type.min().item()
        max_label = edge_type.max().item()
        if min_label < 0 or max_label >= num_classes:
            logging.error(f"Target labels out of range: min={min_label}, max={max_label}, expected range [0, {num_classes - 1}]")
            raise ValueError("Target labels are out of the valid range for cross_entropy.")
        
        loss = F.cross_entropy(preds, edge_type.to(device))
        _, predicted = torch.max(preds, dim=1)
        accuracy = accuracy_score(edge_type.cpu().numpy(), predicted.cpu().numpy())
        f1 = f1_score(edge_type.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        kappa = cohen_kappa_score(edge_type.cpu().numpy(), predicted.cpu().numpy())

    # Prepare evaluation metrics string
    eval_metrics = (
        f"Evaluation Loss: {loss.item()}\n"
        f"Evaluation Accuracy: {accuracy}\n"
        f"F1 Score (Macro): {f1}\n"
        f"Cohen's Kappa: {kappa}\n"
    )
    
    # Write metrics to log_file
    with open(log_file, 'w') as f_log:
        f_log.write(eval_metrics)
    
    # Write metrics to results_file
    with open(results_file, 'w') as f_res:
        f_res.write(eval_metrics)
    
    # Log the metrics
    logging.info(f"Evaluation Loss: {loss.item():.4f}")
    logging.info(f"Evaluation Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Score (Macro): {f1:.4f}")
    logging.info(f"Cohen's Kappa: {kappa:.4f}")

def predict_test_set(model, edge_classifier, edge_index, relation_to_id, output_file, test_df, node_embeddings, device):
    """
    Predict interaction types for the test set and save the predictions.

    :param model: RGCN model
    :param edge_classifier: EdgeClassifier model
    :param edge_index: torch.Tensor, test edge indices
    :param relation_to_id: dict, {relation_name: relation_id, ...}
    :param output_file: str, path to save predictions
    :param test_df: pandas DataFrame, test dataset
    :param node_embeddings: torch.Tensor, [num_nodes, hidden_dim]
    :param device: torch.device
    """

    model.eval()
    edge_classifier.eval()
    predictions = []

    # Reverse mapping from relation IDs to interaction IDs 
    id_to_interaction = {v: int(k.split('_')[1]) for k, v in relation_to_id.items() if 'interaction_' in k}

    with torch.no_grad():
        preds = edge_classifier(node_embeddings, edge_index.to(device))
        _, predicted = torch.max(preds, dim=1)
        predicted = predicted.cpu().numpy()

    for pred in predicted:
        # Map back to original labels 
        interaction_id = id_to_interaction.get(pred, "unknown")
        predictions.append(interaction_id)

    test_df = test_df.copy()
    test_df['Predicted_Y'] = predictions
    test_df.to_csv(output_file, index=False)
    logging.info(f"Test set predictions saved to {output_file}")
