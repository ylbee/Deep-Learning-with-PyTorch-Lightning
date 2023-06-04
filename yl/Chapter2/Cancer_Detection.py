# 1. Importing the packages
import os
import shutil
import opendatasets as od
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchmetrics.functional import accuracy

import pytorch_lightning as pl

print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("torch version:", torch.__version__)
print("pytorch ligthening version:", pl.__version__)


def printProcess(i=0, total=1, des=""):
    # 计算下载百分比
    per = (i + 1) * 100 / total
    # 打印进度条（\r是将光标移动到行的开始，会覆盖上一次打印的内容，形成动态打印）
    print("\r%s:%.f%s" % (des, per, '%'), flush=True, end='')


cancer_labels = pd.read_csv(r'../histopathologic-cancer-detection/train_labels.csv')
print(cancer_labels.head())
print(cancer_labels['label'].value_counts())
print('No. of images in training dataset: ', len(os.listdir(r"../histopathologic-cancer-detection/train")))
print('No. of images in testing dataset: ', len(os.listdir(r"../histopathologic-cancer-detection/test")))

selected_image_list = []
# 判断训练数据集目录是否存在，这个目录存放了在开发时候训练数据集  因为在开发时候数据集不能过大影响速度，从所有数据集中抽取了10000个图片数据
if not os.path.exists(r"../histopathologic-cancer-detection/train_dataset"):
    # 2.2 Downsampling the dataset    downsample the 220,000 images in the train folder to 10,000 images
    # Setting seed to make the results replicable  数据集过大，选取一部分测试训练

    np.random.seed(0)
    train_imgs_orig = os.listdir(r"../histopathologic-cancer-detection/train")

    for img in np.random.choice(train_imgs_orig, 10000):
        selected_image_list.append(img)
    print(len(selected_image_list))

    '''
    fig = plt.figure(figsize=(25, 6))
    for idx, img in enumerate(np.random.choice(selected_image_list, 20)):
        ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
        im = Image.open(r"../histopathologic-cancer-detection/train\\" + img)
        plt.imshow(im)
        lab = cancer_labels.loc[cancer_labels['id'] == img.split('.')[0], 'label'].values[0]
        ax.set_title(f'Label: {lab}')
    '''

    # split the data into train and test data,
    np.random.seed(0)
    np.random.shuffle(selected_image_list)
    cancer_train_idx = selected_image_list[0:8000]
    cancer_test_idx = selected_image_list[8000:]
    print("Number of images in the downsampled training dataset: ", len(cancer_train_idx))
    print("Number of images in the downsampled testing dataset: ", len(cancer_test_idx))
    '''  这里不存到谷歌云盘，改为存到本地电脑
    from google.colab import drive
    drive.mount('/content/gdrive')
    os.mkdir('/content/histopathologic-cancer-detection/train_dataset/')
    for fname in cancer_train_idx:
      src = os.path.join('histopathologic-cancer-detection/train', fname)
      dst = os.path.join('/content/histopathologic-cancer-detection/train_dataset/', fname)
      shutil.copyfile(src, dst)
    print('No. of images in downsampled training dataset: ', len(os.listdir("/content/histopathologic-cancer-detection/train_dataset/")))
    os.mkdir('/content/histopathologic-cancer-detection/test_dataset/')
    for fname in cancer_test_idx:
      src = os.path.join('histopathologic-cancer-detection/train', fname)
      dst = os.path.join('/content/histopathologic-cancer-detection/test_dataset/', fname)
      shutil.copyfile(src, dst)
    print('No. of images in downsampled testing dataset: ', len(os.listdir("/content/histopathologic-cancer-detection/test_dataset/")))
    '''
    os.mkdir(r"../histopathologic-cancer-detection/train_dataset")
    for fname in cancer_train_idx:
        src = os.path.join(r"../histopathologic-cancer-detection/train", fname)
        dst = os.path.join(r"../histopathologic-cancer-detection/train_dataset", fname)
        shutil.copyfile(src, dst)
    print('No. of images in downsampled training dataset: ',
          len(os.listdir(r"../histopathologic-cancer-detection/train_dataset")))

    os.mkdir(r"../histopathologic-cancer-detection/test_dataset")
    for fname in cancer_test_idx:
        src = os.path.join(r"../histopathologic-cancer-detection/train", fname)
        dst = os.path.join(r"../histopathologic-cancer-detection/test_dataset", fname)
        shutil.copyfile(src, dst)
    print('No. of images in downsampled testing dataset: ',
          len(os.listdir(r"../histopathologic-cancer-detection/test_dataset")))
else:  # 已经提取了数据则直接用
    train_dataset = os.listdir(r"../histopathologic-cancer-detection/train_dataset")
    test_dataset = os.listdir(r"../histopathologic-cancer-detection/test_dataset")

    selected_image_list = train_dataset + test_dataset
    # print("数组拼接后",len(selected_image_list))

# Extracting the labels for the images that were selected in the downsampled data
selected_image_labels = pd.DataFrame()
id_list = []
label_list = []

for i, img in enumerate(selected_image_list):
    label_tuple = cancer_labels.loc[cancer_labels['id'] == img.split('.')[0]]
    id_list.append(label_tuple['id'].values[0])
    label_list.append(label_tuple['label'].values[0])
    printProcess(i=i, total=len(selected_image_list), des="正在处理数据" + str(i))

selected_image_labels['id'] = id_list
selected_image_labels['label'] = label_list
print(selected_image_labels.head())


# 2.3 Loading the dataset
class LoadCancerDataset(Dataset):
    def __init__(self, data_folder,
                 transform=T.Compose([T.CenterCrop(32), T.ToTensor()]), dict_labels={}):
        self.data_folder = data_folder
        self.list_image_files = [s for s in os.listdir(data_folder)]
        self.transform = transform
        self.dict_labels = dict_labels
        # self.labels = [dict_labels[i.split('.')[0]] for i in self.list_image_files]

    def __len__(self):
        return len(self.list_image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.list_image_files[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        img_name_short = self.list_image_files[idx].split('.')[0]

        label = self.dict_labels[img_name_short]
        return image, torch.Tensor([label])


# Augmenting the dataset
data_T_train = T.Compose([
    T.CenterCrop(32),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
])
data_T_test = T.Compose([
    T.CenterCrop(32),
    T.ToTensor(),
])

# dictionary with labels and ids of train data
# Load train data
img_label_dict = {k: v for k, v in zip(selected_image_labels.id, selected_image_labels.label)}
train_set = LoadCancerDataset(data_folder=r"../histopathologic-cancer-detection/train_dataset",
                              # datatype='train',
                              transform=data_T_train, dict_labels=img_label_dict)

test_set = LoadCancerDataset(data_folder=r"../histopathologic-cancer-detection/test_dataset",
                             transform=data_T_test, dict_labels=img_label_dict)

batch_size = 256
train_dataloader = DataLoader(train_set, batch_size, num_workers=0, pin_memory=True, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size, num_workers=0, pin_memory=True)


# 3. Building the model
# 31. Initializing the model
class CNNImageClassifier(pl.LightningModule):

    def __init__(self, learning_rate=0.001):
        super().__init__()
        # 学习速率低学习效果好但花更长时间且容易陷入局部最优解，高的学习率可能会减少最初的损失，但永远不会收敛
        self.learning_rate = learning_rate
        # 构建两个卷积层。
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        # 激活函数1
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv_layer2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fully_connected_1 = nn.Linear(in_features=16 * 16 * 6, out_features=1000)
        self.fully_connected_2 = nn.Linear(in_features=1000, out_features=500)
        self.fully_connected_3 = nn.Linear(in_features=500, out_features=250)
        self.fully_connected_4 = nn.Linear(in_features=250, out_features=120)
        self.fully_connected_5 = nn.Linear(in_features=120, out_features=60)
        self.fully_connected_6 = nn.Linear(in_features=60, out_features=1)
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, input):
        output = self.conv_layer1(input)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv_layer2(output)
        output = self.relu2(output)
        output = output.view(-1, 6 * 16 * 16)
        output = self.fully_connected_1(output)
        output = self.fully_connected_2(output)
        output = self.fully_connected_3(output)
        output = self.fully_connected_4(output)
        output = self.fully_connected_5(output)
        output = self.fully_connected_6(output)
        return output

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        print("XOR INPUT:", outputs.shape)
        print("XOR TARGET:", targets.shape)
        train_accuracy = accuracy(preds=outputs, target=targets, task="binary")
        loss = self.loss(outputs, targets)
        self.log('train_accuracy', train_accuracy, prog_bar=True)  # 记录日志
        self.log('train_loss', loss)
        return {"loss": loss, "train_accuracy": train_accuracy}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        test_accuracy = accuracy(preds=outputs, target=targets, task="binary")
        # test_accuracy = self.binary_accuracy(outputs, targets)
        loss = self.loss(outputs, targets)
        self.log('test_accuracy', test_accuracy)
        return {"test_loss": loss, "test_accuracy": test_accuracy}

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=self.learning_rate)
        return optimizer

    # Calculate accuracy for each batch at a time
    def binary_accuracy(self, outputs, targets):
        _, outputs = torch.max(outputs, 1)
        correct_results_sum = (outputs == targets).sum().float()
        acc = correct_results_sum / targets.shape[0]
        return acc

    def predict_step(self, batch, batch_idx):
        return self(batch)


# 32. Configuring the optimizer
# 33. Configuring training and testing
# 5. Training the model
'''
model = CNNImageClassifier()
trainer = pl.Trainer(fast_dev_run=True, gpus=1)
trainer.fit(model, train_dataloaders=train_dataloader)
'''
ckpt_dir = r"../histopathologic-cancer-detection/cnn"
# ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=10)


model = CNNImageClassifier()
trainer = pl.Trainer(default_root_dir=ckpt_dir, log_every_n_steps=25, max_epochs=5)
trainer.fit(model, train_dataloaders=train_dataloader)
# 6. Evaluating the accuracy of the model
trainer.test(model, dataloaders=test_dataloader)

model.eval()

'''
preds = []
for batch_i, (data, target) in enumerate(test_dataloader):
    data, target = data.cuda(), target.cuda()
    output = model.cuda()(data)

    pr = output[:,1].detach().cpu().numpy()
    for i in pr:
        preds.append(i)

test_preds = pd.DataFrame({'imgs': test_set.list_image_files, 'labels':test_set.labels,  'preds': preds})
test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])
test_preds.head()
test_preds['predictions'] = 1
test_preds.loc[test_preds['preds'] < 0, 'predictions'] = 0
test_preds.shape
test_preds.head()
print(len(np.where(test_preds['labels'] == test_preds['predictions'])[0])/test_preds.shape[0])
'''
