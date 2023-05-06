import argparse
import torch
from torch.utils.data import DataLoader
from models.inception_resnet_v1 import InceptionResnetV1
from torchvision import transforms
from torch.optim import Adam
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from losses import CenterLoss, TripletLoss
from sampler import RandomIdentitySampler
from dataset import make_dataset, collate_fn


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
num_classes = 1600
model = InceptionResnetV1(classify=True, num_classes=num_classes).to(device)
batch_size = 16
epochs = 100
lr = 1e-4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="../database/CASIA-WebFace-subset-crop", 
        type=str
    )
    parser.add_argument(
        "--saved_path", default="./checkpoint/base_with_crop_center_triplet.pth", 
        type=str
    )
    parser.add_argument(
        "--use_center_loss", default=True, action="store_true" 
    )
    parser.add_argument(
        "--use_triplet_loss", default=True, action="store_true" 
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    saved_path = args.saved_path
    use_center_loss = args.use_center_loss
    use_triplet_loss = args.use_triplet_loss


    # 加载数据集
    database, train_dataset, test_dataset = make_dataset(name="casia", transform=transform, root=data_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              sampler=RandomIdentitySampler(database.train, batch_size, 4),
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # 新 Loss
    center, triplet = None, None
    if use_center_loss:
        center = CenterLoss(num_classes=num_classes, feat_dim=512)
        print("使用 CenterLoss 进行训练")
    if use_triplet_loss:
        triplet = TripletLoss(0.3)
        print("使用 TripletLoss 进行训练")

    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = 0
        pred_list, label_list = [], []
        model.train()
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits, before_bn_feat = model(images)
            loss = criterion(logits, labels)

            if use_center_loss:
                loss += 0.005 * center(before_bn_feat, labels)
            if use_triplet_loss:
                loss += triplet(before_bn_feat, labels)

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
        if epoch % 5 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                pred_list, label_list = [], []
                for images, labels in test_loader:
                    images = images.to(device)
                    logits, _ = model(images)
                    logits = logits.cpu()
                    pred = torch.argmax(logits, dim=1)
                    pred_list.extend(pred)
                    label_list.extend(labels)
                    loss = criterion(logits, labels)
                    test_loss += loss.item()
            test_acc = accuracy_score(label_list, pred_list)
            print(f'Epoch: {epoch}, Test Loss: {test_loss}, Test Acc: {test_acc:.2%}\n')

    torch.save(model.state_dict(), saved_path)
