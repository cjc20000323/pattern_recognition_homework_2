import argparse
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from models.inception_resnet_v1 import InceptionResnetV1
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

from dataset import make_dataset, collate_fn
from sampler import RandomIdentitySampler


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
model = InceptionResnetV1(classify=False).to(device)
batch_size = 16


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="../database/CASIA-WebFace-subset-crop", 
        type=str
    )
    parser.add_argument(
        "--weight_path", default="./checkpoint/InceptionResnetV1_softmax_casia_crop.pth", 
        type=str
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    weight_path = args.weight_path

    # 读取模型参数
    weight_dict = torch.load(weight_path)
    # 删除分类头的参数
    del weight_dict['logits.weight'], weight_dict['logits.bias']
    # 加载模型参数
    model.load_state_dict(weight_dict)

    # 加载数据集
    database, train_dataset, test_dataset = make_dataset(name="casia", transform=transform, root=data_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              sampler=RandomIdentitySampler(database.train, batch_size, 4),
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    # 训练集的embeddings
    train_embeddings, train_labels = [], []
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        embeddings, _ = model(images)
        embeddings = embeddings.cpu()
        train_embeddings.append(embeddings)
        train_labels.append(labels)
    train_embeddings = torch.cat(train_embeddings)
    train_labels = torch.cat(train_labels)

    # 测试集的embeddings
    test_embeddings, test_labels = [], []
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        embeddings, _ = model(images)
        embeddings = embeddings.cpu()
        test_embeddings.append(embeddings)
        test_labels.append(labels)
    test_embeddings = torch.cat(test_embeddings)
    test_labels = torch.cat(test_labels)

    # 用KNN匹配特征
    print("默认距离")
    knn = KNeighborsClassifier(1)
    knn.fit(train_embeddings, train_labels)
    acc = knn.score(test_embeddings, test_labels)
    print(f'Test Acc: {acc:.2%}')

    print("\n余弦距离")
    knn = KNeighborsClassifier(1, metric="cosine")
    knn.fit(train_embeddings, train_labels)
    acc = knn.score(test_embeddings, test_labels)
    print(f'Test Acc: {acc:.2%}')


if __name__ == '__main__':
    main()
