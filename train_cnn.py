import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
# 定義簡單的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 计算卷积层输出大小
        self.fc_input_size = self._get_fc_input_size()
        self.fc = nn.Linear(self.fc_input_size, self.fc_input_size)
        self.fc1 = nn.Linear(self.fc_input_size, num_classes)

        # self.fc1 = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # print(x.size(0))
        x = self.fc(x)
        x = self.fc1(x)
        return x

    def _get_fc_input_size(self):
        # 辅助函数用于计算卷积层输出大小
        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)
            return x.view(1, -1).size(1)
# 自定義 Dataset 類別
# Assuming that each image has a corresponding label (0 or 1)
# If you have the actual labels, you should load them from your dataset
# and convert them to a tensor.

# For example, if your dataset structure is like:
# ./cnn_trainset/class_0/...
# ./cnn_trainset/class_1/...

# 修改 CustomDataset 类中的 __getitem__ 方法
class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.classes = os.listdir(data_folder)
        self.image_paths = [os.path.join(data_folder, cls, img) for cls in self.classes for img in
                            os.listdir(os.path.join(data_folder, cls))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Extract class from the path
        label = int(img_path.split(os.sep)[-2])

        # 将非这两个类别的人标记为2（其他）
        if label not in [0, 1]:
            label = 2

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label



# 圖像預處理和轉換
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# 訓練參數
num_epochs = 100
batch_size = 32
learning_rate = 0.001
num_classes = 2  # 兩個類別，你需要根據實際情況修改

# 訓練資料夾路徑
train_data_folder = './trainset'

# 創建訓練 Dataset 和 DataLoader
train_dataset = CustomDataset(train_data_folder, transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、損失函數和優化器
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 訓練迴圈
for epoch in tqdm(range(num_epochs), desc='Training', unit='epoch'):
    for images, labels in train_loader:
        # 清除梯度
        optimizer.zero_grad()

        # 正向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'iutrump2.pth')


# 儲存訓練好的模型
# torch.save(model.state_dict(), 'iutrump2.pth')
print('Model trained and saved successfully.')
