import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


def plot_loss(train_losses, val_losses):
    """ Plot training and validation loss """
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', linestyle='-', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def train(
        model, criterion, optimizer, scheduler,
        dataloaders_dict, dataset_sizes_dict, device,
        num_epochs=25, writer=None, hyperparameters=None,
        model_save_path=None
):
    """ Train the model """
    start_time = time.time()
    ## Phase 1: Initialize Training
    # Saves best model weights (useful for early stopping)
    # Tracks training/validation losses
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    best_f1_score = 0.0

    train_losses = []
    val_losses = []

    if hyperparameters and writer:
        for key, value in hyperparameters.items():
            writer.add_text(f'Hyperparameters/{key}', str(value))

    for epoch in range(num_epochs):
        logger.info('\n')
        logger.info(f'Epoch {epoch}/{num_epochs-1}')
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        epoch_train_f1 = 0.0
        epoch_validation_f1 = 0.0

        for phase in ['train', 'validation']:
            ## Phase 2: Training & Validation Loop
            # in train mode, we ENABLE GRADIENT UPDATES
            # in eval mode, we DISABLE DROPOUT & BATCH NORM UPDATES
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            all_predictions = []
            all_labels = []

            # 1. Loads a batch of images
            # 2. Moves tensors to GPU
            # 3. Clears previous gradients
            for inputs, labels in tqdm(
                    dataloaders_dict[phase], desc=f'{phase} epoch {epoch}'
            ):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # 1. Performs forward pass
                # 2. Calculates loss using CrossEntropyLoss
                # 3. Updates weights in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    predicted_outputs = model(inputs)
                    _, predictions = torch.max(predicted_outputs, 1)
                    loss = criterion(predicted_outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            ## Phase 3: Performance Metrics
            # Computes loss for the epoch
            # Calculates accuracy & F1-score (better metric for imbalanced classes)
            epoch_loss = running_loss / dataset_sizes_dict[phase]
            epoch_accuracy = running_corrects.double() / dataset_sizes_dict[phase]

            all_predictions = np.concatenate(all_predictions)
            all_labels = np.concatenate(all_labels)

            if phase == 'train':
                epoch_train_f1 = f1_score(
                    all_labels, all_predictions, average='macro' # Weighted F1 Score
                )
            if phase == 'validation':
                epoch_validation_f1 = f1_score(
                    all_labels, all_predictions, average='macro' # Weighted F1 Score
                )

            if phase == 'train':
                scheduler.step(epoch_train_f1)
                epoch_train_loss = epoch_loss

                # Calculate Precision and Recall
                epoch_train_precision = precision_score(
                    all_labels, all_predictions, average='macro', zero_division=1
                )
                epoch_train_recall = recall_score(
                    all_labels, all_predictions, average='macro', zero_division=1
                )

                logger.info(
                    f"Train Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_accuracy:.4f} - "
                    f"Train F1: {epoch_train_f1:.4f} - Train Precision: {epoch_train_precision:.4f} - "
                    f"Train Recall: {epoch_train_recall:.4f}"
                )
                if writer:
                    writer.add_scalar('Loss/train', epoch_train_loss, epoch)
                    writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)
                    writer.add_scalar('F1/train', epoch_train_f1, epoch)
                    writer.add_scalar('Precision/train', epoch_train_precision, epoch)
                    writer.add_scalar('Recall/train', epoch_train_recall, epoch)

                print(f"\nTrain Epoch {epoch}\nTrain F1 Score: {epoch_train_f1:.4f} - Train Precision: {epoch_train_precision:.4f} - Train Recall: {epoch_train_recall:.4f}\n")

            if phase == 'validation':
                epoch_val_loss = epoch_loss

                # Calculate Precision and Recall
                epoch_validation_precision = precision_score(
                    all_labels, all_predictions, average='macro', zero_division=1
                )
                epoch_validation_recall = recall_score(
                    all_labels, all_predictions, average='macro', zero_division=1
                )

                logger.info(
                    f"Validation Loss: {epoch_loss:.4f} - Validation Accuracy: {epoch_accuracy:.4f} - "
                    f"Validation F1: {epoch_validation_f1:.4f} - "
                    f"Validation Precision: {epoch_validation_precision:.4f} - "
                    f"Validation Recall: {epoch_validation_recall:.4f}"
                )

                if writer:
                    writer.add_scalar('Loss/validation', epoch_val_loss, epoch)
                    writer.add_scalar('Accuracy/validation', epoch_accuracy, epoch)
                    writer.add_scalar('F1/validation', epoch_validation_f1, epoch)
                    writer.add_scalar('Precision/validation', epoch_validation_precision, epoch)
                    writer.add_scalar('Recall/validation', epoch_validation_recall, epoch)

                print(f"\nValidation Epoch {epoch}\nValidation F1 Score: {epoch_validation_f1:.4f} - Validation Precision: {epoch_validation_precision:.4f} - Validation Recall: {epoch_validation_recall:.4f}\n")

                # Saves the best model (based on weighted validation F1 score)
                if epoch_validation_f1 > best_f1_score:
                    best_f1_score = epoch_validation_f1
                    best_model_weights = copy.deepcopy(model.state_dict())
                    if model_save_path:
                        torch.save(model.state_dict(), model_save_path)
                        logger.info(
                            f"Best model saved at epoch {epoch} with weighted F1 score {epoch_validation_f1:.4f}")



                # Saves the best model (based on validation accuracy)
                #if epoch_accuracy > best_accuracy:
                #    best_accuracy = epoch_accuracy
                #    best_model_weights = copy.deepcopy(model.state_dict())
                #    if model_save_path:
                #        torch.save(model.state_dict(), model_save_path)
                #        logger.info(f"Best model saved at epoch {epoch} with accuracy {epoch_accuracy:.4f}")

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

    time_elapsed = time.time() - start_time
    logger.info(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    logger.info(f'Best Validation F1 Score: {best_f1_score:.4f}')

    model.load_state_dict(best_model_weights)

    # plot_loss(train_losses, val_losses)

    return model


def evaluate(model, test_dataloader, criterion,
             device, class_categories, writer=None):
    """ Evaluate the trained model """
    num_classes = len(class_categories)
    model.eval()
    loss_on_test_dataset = 0.0
    correct_class = [0 for _ in range(num_classes)]
    total_correct_for_all_classes = [0 for _ in range(num_classes)]

    for test_inputs, test_labels in tqdm(test_dataloader):
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

        with torch.no_grad():
            output_test = model(test_inputs)
            loss = criterion(output_test, test_labels)

        loss_on_test_dataset += loss.item() * test_inputs.size(0)
        _, pred_test = torch.max(output_test, 1)

        correct_tensor = pred_test.eq(test_labels.view_as(pred_test))
        correct = correct_tensor.cpu().numpy()

        for i in range(len(test_labels)):
            label = test_labels.data[i]
            correct_class[label] += correct[i].item()
            total_correct_for_all_classes[label] += 1

    loss_on_test_dataset /= len(test_dataloader.dataset)
    logger.info(f'Test Loss: {loss_on_test_dataset:.4f}')

    if writer:
        writer.add_scalar('Loss/test', loss_on_test_dataset)

    for i in range(num_classes):
        accuracy = 100 * correct_class[i] / total_correct_for_all_classes[i]
        logger.info(f'Test Accuracy of {class_categories[i]}: {accuracy:.2f}%')

        # if writer:
        #     writer.add_scalar(f'Accuracy/test_{class_categories[i]}', accuracy)

    overall_accuracy = 100 * np.sum(correct_class) / np.sum(total_correct_for_all_classes)
    logger.info(f'Overall Test Accuracy: {overall_accuracy:.2f}%')

    if writer:
        writer.add_scalar('Accuracy/overall_test', overall_accuracy)