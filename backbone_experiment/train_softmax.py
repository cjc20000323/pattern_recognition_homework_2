import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from models.inception_resnet_v1 import InceptionResnetV1
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from models.iresnet import iresnet50
from models.mobile_facenet import MobileFacenet


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
data_dir = r"../dataset/CASIA-WebFace-subset"
# model = InceptionResnetV1(classify=True, num_classes=1000).to(device)
model = iresnet50(classify=True, num_classes=1000).to(device)
# model = MobileFacenet(classify=True, num_classes=1000).to(device)
batch_size = 16
epochs = 100
lr = 1e-4
saved_path = 'checkpoint/iresnet50_webface_softmax_nocrop.pth'


if __name__ == '__main__':
    # 加载数据集，划分训练集和测试集
    dataset = ImageFolder(root=data_dir, transform=transform)
    idx = range(len(dataset))
    train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=dataset.targets, random_state=0)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = 0
        pred_list, label_list = [], []
        model.train()
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu().item()
            pred = torch.argmax(logits.detach().cpu(), dim=1)
            pred_list.extend(pred)
            label_list.extend(labels.detach().cpu())
        train_acc = accuracy_score(label_list, pred_list)
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Train Acc: {train_acc:.2%}')

        # 测试，实际测试集无法看到，这里仅供参考
        model.eval()
        test_loss = 0
        with torch.no_grad():
            pred_list, label_list = [], []
            for images, labels in tqdm(test_loader):
                images = images.to(device)
                logits = model(images).cpu()
                pred = torch.argmax(logits, dim=1)
                pred_list.extend(pred)
                label_list.extend(labels)
                loss = criterion(logits, labels)
                test_loss += loss.item()
        test_acc = accuracy_score(label_list, pred_list)
        print(f'Epoch: {epoch}, Test Loss: {test_loss}, Test Acc: {test_acc:.2%}\n')

    torch.save(model.state_dict(), saved_path)
