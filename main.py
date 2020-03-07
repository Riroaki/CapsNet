import torch
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
from capsnet import CapsNet, CapsuleLoss
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    """Wrap of MNIST dataset from torchvision."""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


def main():
    torch.autograd.set_detect_anomaly(True)
    # Load model
    model = CapsNet()
    caps_loss = CapsuleLoss()
    optimizer = Adam(model.parameters())

    # Load data
    dataset = MNIST(root='./data', download=True)
    DATA_SPLIT = [50000, 10000]
    train_data, test_data = random_split(dataset.data.float().unsqueeze(dim=1), DATA_SPLIT)
    train_label, test_label = random_split(dataset.targets, DATA_SPLIT)
    train_loader = DataLoader(dataset=MNISTDataset(train_data, train_label),
                              batch_size=100,
                              num_workers=4)
    test_loader = DataLoader(dataset=MNISTDataset(test_data, test_label),
                             batch_size=100,
                             num_workers=4)

    # Train
    EPOCHES = 10
    model.train()
    for ep in range(EPOCHES):
        train_bar = tqdm(total=500)
        optimizer.zero_grad()
        total_loss = 0.
        for images, labels in train_loader:
            # Categogrical encoding
            labels = torch.eye(10).index_select(dim=0, index=labels)
            loss = caps_loss(images, labels, *model(images))
            total_loss += loss
            loss.backward()
            optimizer.step()
            train_bar.update()
        print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))
        train_bar.close()

    # Eval
    model.eval()
    correct, total = 0, 0
    eval_bar = tqdm(total=100)
    for images, labels in test_loader:
        logits, reconstructions = model(images)
        pred_labels = torch.argmax(logits, dim=1)
        correct += torch.sum(pred_labels == labels).item()
        total += len(labels)
        eval_bar.update()
    print('Accuracy: {}'.format(correct / total))
    eval_bar.close()

    # Save model
    torch.save(model.state_dict(), './model/capsnet.pt')


if __name__ == '__main__':
    main()
