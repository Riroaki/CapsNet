import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from capsnet import CapsNet, CapsuleLoss
from torch.utils.data import TensorDataset


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    model = CapsNet().to(device)
    criterion = CapsuleLoss()
    optimizer = optim.Adam(model.parameters())

    # Load data
    TRAIN_SIZE, TEST_SIZE = 50000, 10000
    dataset = MNIST(root='./data', download=True)
    random_indices = torch.randperm(60000)
    BATCH_SIZE = 100
    train_loader = DataLoader(
        dataset=TensorDataset(
            dataset.data.index_select(dim=0, index=random_indices[:TRAIN_SIZE]),
            dataset.targets.index_select(dim=0, index=random_indices[:TRAIN_SIZE])),
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True)
    test_loader = DataLoader(
        dataset=TensorDataset(
            dataset.data.index_select(dim=0, index=random_indices[TRAIN_SIZE:]),
            dataset.targets.index_select(dim=0, index=random_indices[TRAIN_SIZE:])),
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True)

    # Train
    EPOCHES = 100
    model.train()
    for ep in range(EPOCHES):
        optimizer.zero_grad()
        total_loss = 0.
        batch_id = 1
        for images, labels in train_loader:
            # Add channels = 1
            images = (images.float() / 255.0).unsqueeze(dim=1).to(device)
            # Categogrical encoding
            labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
            logits, reconstruction = model(images, labels)
            # Compute loss & accuracy
            loss = criterion(images, labels, logits, reconstruction)
            accuracy = torch.sum(
                torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).item() / len(images)
            total_loss += loss
            loss.backward()
            optimizer.step()
            print('Epoch {}, batch {}: {}, accuracy: {}'.format(ep + 1,
                                                                batch_id,
                                                                loss.item(),
                                                                accuracy))
            batch_id += 1
        print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))

    # Eval
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        # Add channels = 1
        images = (images.float() / 255.0).unsqueeze(dim=1).to(device)
        # Categogrical encoding
        labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
        logits, reconstructions = model(images, labels)
        pred_labels = torch.argmax(logits, dim=1)
        correct += torch.sum(pred_labels == torch.argmax(labels, dim=1)).item()
        total += len(labels)
    print('Accuracy: {}'.format(correct / total))

    # Save model
    torch.save(model.state_dict(), './model/capsnet_ep{}_acc{}.pt'.format(EPOCHES, correct / total))


if __name__ == '__main__':
    main()
