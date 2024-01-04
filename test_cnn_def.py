import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 计算卷积层输出大小
        self.fc_input_size = self._get_fc_input_size()

        self.fc1 = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
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

# ... (與原始程式碼一致)

def predict_face_class(test_image):
    # 讀取測試圖片
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load('cagetrump.pth'))
    model.eval()
    test_image_tensor = transform(test_image).unsqueeze(0)
    cf = 10000000000
    # 進行臉部分類預測
    with torch.no_grad():
        output = model(test_image_tensor)


    # 獲取預測結果
    cf, predicted_class = torch.max(output, 1)
    lable = predicted_class.item()

    return float(lable[0]),float(cf[0])


# 圖像預處理和轉換

if __name__ == "__main__":
    # 測試圖片路徑
    test_image_path = './cnn_testset/9.jpg'
    # 使用函式進行預測
    lable, cf = predict_face_class(test_image_path)
    # 根據預測結果進行後續處理
    print('lable:',lable,"cf:",cf)

