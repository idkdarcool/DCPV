import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

def train_feature_extractor(model, train_loader, device, epochs=5):
    """
    Train the feature extractor (CNN) on the full training set using CrossEntropy.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Training Feature Extractor...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}, Acc {correct/total:.4f}")

    return model

def train_mfvi(bayes_layer, features, targets, epochs=100, lr=0.01):
    """
    Train MFVI layer using ELBO optimization (Minimize Negative ELBO).
    Cost = NLL + KL Divergence.
    """
    optimizer = optim.Adam(bayes_layer.parameters(), lr=lr)
    
    bayes_layer.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward (Mean)
        pred_mean = bayes_layer(features, sample=False)
        
        # Expected Log Likelihood (Closed Form approximation)
        mse = torch.sum((targets - pred_mean) ** 2)
        
        # Trace term: sum((X^2) @ (Sigma^2))
        std = torch.exp(bayes_layer.weight_log_sigma)
        var_weights = std ** 2
        trace_term = torch.sum((features ** 2) @ var_weights)
        
        nll = 0.5 * (mse + trace_term)
        kl = bayes_layer.kl_divergence()
        
        loss = nll + kl
        
        loss.backward()
        optimizer.step()
        
    return bayes_layer

def train_classifier(model, train_loader, val_loader, device, epochs=50, lr=1e-3, use_wandb=False):
    """Standard classifier training loop with WANDB logging."""
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_loss": avg_val_loss,
                "val_acc": val_acc
            })
            
        if val_acc > best_acc:
            best_acc = val_acc
            
    return model, best_acc
