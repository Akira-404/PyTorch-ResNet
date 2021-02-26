import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import resnet34


def train():
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # 训练集数据地址
    image_path = "/home/lee/pyCode/dl_data/flower_photos"
    assert os.path.exists(image_path), "{} 路径不存在.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    print("classes:", train_dataset.classes)  # 根据分类文件夹的名字来确定的类别
    print("class_to_idx:", train_dataset.class_to_idx)  # 按顺序为这些类别定义索引为0,1...

    # 将类型和索引反序并保持到json文件
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    print("cla_dict:", cla_dict)

    # 将python对象编码成Json字符串
    json_str = json.dumps(cla_dict, indent=4)
    # 写入文件
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32

    # os.cpu_count()Python中的方法用于获取系统中的CPU数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True,
                              num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = DataLoader(validate_dataset,
                                 batch_size=batch_size, shuffle=False,
                                 num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    isTrain =True
    if isTrain:
        # 5分类，使用辅助分类器，初始化权重
        net=resnet34()
        net.to(device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0003)

        epochs = 30
        best_acc = 0.0
        save_path = "./model_data.pth"
        if os.path.exists(save_path):
            os.mkdir(save_path)

        train_steps = len(train_loader)

        for epoch in range(epochs):
            # train
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()

                logits, aux_logits2, aux_logits1 = net(images.to(device))

                loss0 = loss_function(logits, labels.to(device))
                loss1 = loss_function(aux_logits1, labels.to(device))
                loss2 = loss_function(aux_logits2, labels.to(device))
                loss = loss0 + loss1 * 0.3 + loss2 * 0.3
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch

            # 不跟踪梯度
            with torch.no_grad():
                val_bar = tqdm(validate_loader, colour='green')

                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))  # eval model only have last output layer
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            #只保存最好的一次结果
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

            print('训练完成')


if __name__ == "__main__":
    train()