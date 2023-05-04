import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from models.inception_resnet_v1 import InceptionResnetV1
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
data_dir = r"dataset/faces96-crop"
model = InceptionResnetV1(classify=False).to(device)
batch_size = 16
weight_path = 'checkpoint/InceptionResnetV1_softmax.pth'


@torch.no_grad()
def main():
    # 读取模型参数
    weight_dict = torch.load(weight_path)
    # 删除分类头的参数
    del weight_dict['logits.weight'], weight_dict['logits.bias']
    # 加载模型参数
    model.load_state_dict(weight_dict)

    # 加载数据集，划分训练集和测试集（与训练阶段划分一致）
    dataset = ImageFolder(root=data_dir, transform=transform)
    idx = range(len(dataset))
    train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=dataset.targets, random_state=0)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))
    # 训练集的embeddings
    train_embeddings, train_labels = [], []
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        embeddings = model(images).cpu()
        train_embeddings.append(embeddings)
        train_labels.append(labels)
    train_embeddings = torch.concatenate(train_embeddings)
    train_labels = torch.concatenate(train_labels)

    # 测试集的embeddings
    test_embeddings, test_labels = [], []
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        embeddings = model(images).cpu()
        test_embeddings.append(embeddings)
        test_labels.append(labels)
    test_embeddings = torch.concatenate(test_embeddings)
    test_labels = torch.concatenate(test_labels)

    # 用KNN匹配特征
    knn = KNeighborsClassifier(1)
    knn.fit(train_embeddings, train_labels)
    acc = knn.score(test_embeddings, test_labels)
    print(f'Test Acc: {acc:.2%}')


if __name__ == '__main__':
    main()
