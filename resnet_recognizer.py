import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

def MTCNN_detector(data_path):

    workers = 0 if os.name == 'nt' else 4

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print('在该设备上运行: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    def collate_fn(x):
        return x[0]

    dataset = datasets.ImageFolder(data_path)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    aligned = []
    names = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('检测到的人脸及其概率: {:8f}'.format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    return aligned,names

def resnet_img_recognizer(aligned,names=[''],model='resnet_recognizer.pth'):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('在该设备上运行: {}'.format(device))
    aligned = torch.stack(aligned).to(device)

    #resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    #resnet = InceptionResnetV1(pretrained='vggface2',classify=True).eval().to(device)
    resnet = torch.load(model).eval().to(device)
    embeddings = resnet(aligned).detach().cpu()
    # print(embeddings)

    # dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    # print(pd.DataFrame(dists, columns=names, index=names))

    [score,lable] = torch.max(embeddings, 1)
    # print(torch.max(embeddings,1))

    return float(score[0]),float(lable[0])



def train_resnet_recognizer(train_path,test_path,model='resnet_recognizer.pth'):

    batch_size = 32
    epochs = 10
    workers = 0 if os.name == 'nt' else 8

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(format(device))

    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    train_dataset = datasets.ImageFolder(train_path, transform=trans)
    test_dataset = datasets.ImageFolder(test_path, transform=trans)

    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(train_dataset.class_to_idx)
    ).to(device)

    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])

    train_loader = DataLoader(
        train_dataset,
        num_workers=workers,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        test_dataset,
        num_workers=workers,
        batch_size=batch_size,
        shuffle=True
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print('\n\n初始化')
    print('-' * 10)
    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    for epoch in range(epochs):
        print('\nepoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        resnet.train()
        training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        resnet.eval()
        training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

    writer.close()
    torch.save(resnet,'./'+model)

def test_resnet_recognizer(test_path,save_path, model='resnet_recognizer.pth'):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(format(device))

    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    count = 0
    T_count = 0
    for lable_name in tqdm(os.listdir(test_path)):
        for img_name in tqdm(os.listdir(test_path+'/'+lable_name)):

            img = Image.open(test_path+'/'+lable_name+'/'+img_name)
            img = img.convert('RGB')
            img = trans(img)
            resnet = torch.load(model).eval().to(device)
            img = torch.stack([img]).to(device)
            predict = resnet(img)
            #print(predict)
            [score, lable] = torch.max(predict, 1)
            #print(torch.max(predict, 1))

            if os.listdir(test_path)[lable] == lable_name:
                T_count += 1
            count += 1

            img = cv2.imread(test_path + '/' + lable_name + '/' + img_name)
            if lable == 0:
                cv2.putText(img, 'iu: ' + str(np.round(float(score),2)), (5,img.shape[0]-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)
            elif lable == 1:
                cv2.putText(img, 'trump: ' + str(np.round(float(score),2)), (5,img.shape[0]-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)
            cv2.imwrite(save_path + '/' + lable_name + '/' + img_name,img)

    acc = T_count/count
    print('acc: ',acc)


if __name__ == '__main__':

    # test_path = './resnet_test_pict'
    # [aligned,names] = MTCNN_detector(test_path)
    # print(aligned)
    # print(aligned[0].shape)
    # print(names)

    # img1 = Image.open('./test_face_pict/cage/847201438.jpg').resize((160,160))
    # if img1.mode == 'L':
    #     img1 = img1.convert('RGB')
    # img1 = np.array(img1)
    # img1 = torch.from_numpy(img1 / 255.0).float()
    # img1 = img1.permute(2, 0, 1)
    #
    # img2 = Image.open('./test_face_pict/trump/888902682.jpg').resize((160, 160))
    # if img2.mode == 'L':
    #     img2 = img2.convert('RGB')
    # img2 = np.array(img2)
    # img2 = torch.from_numpy(img2 / 255.0).float()
    # img2 = img2.permute(2, 0, 1)
    #
    # names = ['cage1']
    # aligned = [img1]
    #
    # names = ['cage1','trump1']
    # aligned = [img1,img2]
    #
    # print(aligned)
    # print(aligned[0].shape)
    # print(names)
    #
    # print(resnet_recognizer(aligned,names))

    # train_path = './data/train'
    # test_path = './data/test'
    # train_resnet_recognizer(train_path,test_path,model='resnet_recognizer.pth')

    # test_path = './data/test'
    # save_path = './result/test'
    # test_path = './data/crop_test_L'
    # save_path = './result/test_L/crop_pic'
    # test_path = './data/test_L'
    # save_path = './result/test_L/full_pic'
    # test_path = './data/crop_test_E'
    # save_path = './result/test_E/crop_pic'
    test_path = './data/test_E'
    save_path = './result/test_E/full_pic'
    test_resnet_recognizer(test_path,save_path,model='resnet_recognizer.pth')