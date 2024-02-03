import torch
import torchvision
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from pix2pix_train import GtsrbDataset


class VGGNet_19(nn.Module):
    def __init__(self, num_classes=43):  # num_classes，此处为 二分类值为2
        super(VGGNet_19, self).__init__()
        net = torchvision.models.vgg19(pretrained=True)  # 从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG19的特征层
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), -1)
        # print(x.size)
        x = self.classifier(x)
        return x


class ResNet_32(nn.Module):
    def __init__(self, num_classes=43):  # num_classes，此处为 二分类值为2
        super(ResNet_32, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=True)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class Googlenet(nn.Module):
    def __init__(self, num_classes=43):  # num_classes，此处为 二分类值为2
        super(Googlenet, self).__init__()
        self.model = torchvision.models.googlenet(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    # -------------------------1. prepare data-------------------------
    train_loader = DataLoader(GtsrbDataset(" "), batch_size=16, shuffle=True)
    val_loader = DataLoader(GtsrbDataset(" "), batch_size=16, shuffle=True)
    test_loader = DataLoader(GtsrbDataset(" "), batch_size=16, shuffle=True)

    # -------------------------2. load model-------------------------
    model = VGGNet_19(num_classes=43)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # # -------------------------3. prepare super parameters-------------------------
    # criterion = nn.CrossEntropyLoss()
    # learning_rate = 1e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler_G = StepLR(optimizer, step_size=2, gamma=0.2)
    #
    # # ---------------------------------4. train--------------------------------
    # val_acc_list = []
    # model.train()
    # for epoch in range(10):
    #     model.train()
    #     train_loss = 0.0
    #     correct = 0
    #     total = 0
    #     for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    #         data, target = data.to(device), target.to(device)
    #         optimizer.zero_grad()
    #         output = model(data)
    #
    #         _, predicted = torch.max(output.data, dim=1)
    #         total += target.size(0)
    #         correct += (predicted == target).sum().item()
    #
    #         loss = criterion(output, target)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #
    #     # val
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for batch_idx, (data, target) in enumerate(val_loader):
    #             data, target = data.to(device), target.to(device)
    #             output = model(data)
    #             _, predicted = torch.max(output.data, dim=1)
    #             total += target.size(0)
    #             correct += (predicted == target).sum().item()
    #         acc_val = correct / total
    #         val_acc_list.append(acc_val)
    #
    #     # save model
    #     torch.save(model.state_dict(), "./LISA/Vgg19-last.pt")
    #     if acc_val == max(val_acc_list):
    #         torch.save(model.state_dict(), "./LISA/Vgg19-best.pt")
    #         print("save epoch {} model".format(epoch))
    #     print("epoch = {},  loss = {},  acc_val = {}".format(epoch, train_loss, acc_val))

    #  -------------------test model-------------------------
    # model.load_state_dict(torch.load("D:\\Study\\wlh\\My_pix2pix\\saved_models\\lisa\\Vgg19\\best.pt"))
    # model = model.cuda()
    # model.eval()
    #
    # total = 0
    # correct = 0
    #
    # for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
    #     data, target = data.to(device), target.to(device)
    #     output = model(data)
    #     _, predicted = torch.max(output.data, dim=1)
    #     total += target.size(0)
    #     correct += (predicted == target).sum().item()
    # acc_test = correct / total
    #
    # print("测试集准确率：%.5f" % acc_test)