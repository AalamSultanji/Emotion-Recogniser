import torch
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs = 50):
    '''
    Training function for the model.
    Arguments: 
    '''
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    os.makedirs('weights', exist_ok=True)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)
        #validation part
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Epoch {epoch+1}/{epochs} || Train Loss: {avg_train_loss:.4f} || Train Acc: {train_accuracy:.4f} || Val Loss: {val_loss:.4f} || Val Acc: {val_accuracy:.4f}')

        #saving best  model 
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'weights/best_model.pth')
        
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies
    }
    return model, history

def evaluate(model, loader, criterion, device):
    '''
    Evaluation for the model
    '''
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def test(model, test_loader, device, class_names):
    '''
    Testing function for the model, computes the confusion matrix and classification report. 
    '''
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc_score = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Test Accuracy: {acc_score:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print("Classification Report:")
    class_report = classification_report(all_labels, all_preds, target_names=class_names)
    print(class_report)
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return {
        'accuracy': acc_score,
        'f1_score': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'confusion_matrix': cm
    }

