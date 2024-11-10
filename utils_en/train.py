import os
import csv
import time
import torch
from torch import nn
from DGCNN import DGCNN
from torch.utils.data import DataLoader
from dataloader import ModelNet400Dataset


def train():

    if not os.path.exists('../model'):
        os.mkdir('../model')

    batch_size = 32
    epochs = 50
    ndf = 64
    k = 10
    num_classes = 40
    best_acc = 0

    train_dataset = ModelNet400Dataset(data_dir='data/modelnet40_hdf5_2048', split='train')
    test_dataset = ModelNet400Dataset(data_dir='data/modelnet40_hdf5_2048', split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    print("Finish Loading Data")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = DGCNN(ndf, k, num_classes).to(device)
    print("Finished load model")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_csv = '../train.csv'

    with open(train_csv, mode='w', newline="") as f:
        fieldnames = ['Epoch', "Train_loss", "Test_acc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print("Start Training")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for points, labels in test_loader:
                points, labels = points.to(device), labels.to(device)
                outputs = model(points)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total

        with open(train_csv, mode='a', newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({"Epoch": epoch+1, 'Train_loss': train_loss, "Test_acc": test_acc})

        print(f"Epoch: {epoch+1}, Loss: {train_loss:.2f}, Accuracy: {test_acc*100:.2f}%, Time: {time.time() - start_time:.2f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'../model/best_model_{best_acc*100:.2f}%.pt')
            print(f"Best model accuracy: {best_acc*100:.2f}%")


if __name__ == '__main__':
    train()


