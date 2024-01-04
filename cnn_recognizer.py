import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
# 0 -> iu 1 -> trump
test_folder_path = './your_testdataset_path/0or1/'
results_folder = './result/your_result_path / 0 or 1 /'
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
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
        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)
            return x.view(1, -1).size(1)
transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(),])

model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load('cnn_recognizer.pth'))
model.eval()

image_files = [f for f in os.listdir(test_folder_path) if f.endswith('.jpg') or f.endswith('.png')]
flag0 = 0
flag1 = 0
cntall = 0

for image_file in image_files:
    cntall += 1
    image_path = os.path.join(test_folder_path, image_file)
    im = cv2.imread(image_path)
    test_image = Image.open(image_path).convert('RGB')
    test_image_tensor = transform(test_image).unsqueeze(0)
    with torch.no_grad():
        output = model(test_image_tensor)
    cf, predicted_class = torch.max(output, 1)
    text0 = f'iu: {cf.item():.2f}'
    text1 = f'trump: {cf.item():.2f}'
    if predicted_class.item() == 0:
        flag0 += 1
        cv2.putText(im, text0, (5, im.shape[0]-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    elif predicted_class.item() == 1:
        flag1 += 1
        cv2.putText(im, text1, (5, im.shape[0]-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    result_image_path = os.path.join(results_folder, image_file)
    cv2.imwrite(result_image_path, im)
print("iu:", flag0)
print("trump:", flag1)
print("all number", cntall)

