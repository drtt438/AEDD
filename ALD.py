"""
    在干净样本上训练生成对抗网络，
    当对抗补丁引入后，对抗样本所在区域会变的难以生成，
    导致生成图像和输入图像在对抗补丁所在区域产生较大的异常损失，
    尝试以此方法定位对抗补丁位置
"""
import math

import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from target_models import VGGNet_19


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################################
#           U-NET----generator of pix2pix
##############################################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)  # 64
        d2 = self.down2(d1)  # 128
        d3 = self.down3(d2)  # 256
        d4 = self.down4(d3)  # 512
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)  # 1024
        u2 = self.up2(u1, d6)  # 1024
        u3 = self.up3(u2, d5)  # 1024
        u4 = self.up4(u3, d4)  # 1024
        u5 = self.up5(u4, d3)  # 512
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            # *discriminator_block(7, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


def comput_IoU(mask_predict, mask_x, threshold):

    mask_predict = torch.where(mask_predict < threshold, 0, 1)
    mask_predict = torch.sum(mask_predict, dim=1)
    mask_predict = torch.where(mask_predict != 0, 1, 0)

    mask_x = torch.sum(mask_x, dim=1)
    mask_x = torch.where(mask_x != 0, 1, 0)

    iou = 0
    for i in range(5):
        result = mask_x[i] + mask_predict[i]
        Intersection = (result == 2).sum().item()
        Union = (result != 0).sum().item()
        iou += Intersection / Union
    return iou / 5


def comput_Coverage(mask_predict, mask_x, threshold):

    mask_predict = torch.where(mask_predict < threshold, 0, 1)
    mask_predict = torch.sum(mask_predict, dim=1)
    mask_predict = torch.where(mask_predict != 0, 1, 0)

    mask_x = torch.sum(mask_x, dim=1)
    mask_x = torch.where(mask_x != 0, 1, 0)

    Coverage = 0
    for i in range(5):
        original = (mask_x[i] == 1).sum().item()
        result = mask_x[i] + mask_predict[i]
        cover = (result == 2).sum().item()
        Coverage += cover / original
    return Coverage / 5


if __name__ == '__main__':
    generator = GeneratorUNet()

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator = generator.cuda()
    # # ---------------------------------训练好的pix2pix网络------------------------------
    generator.load_state_dict(torch.load(" "))
    generator.eval()

    # # ---------------------------------加载对抗补丁------------------------------
    ori_y_path = ["D:\\Study\\wlh\\adversarial_patch\\PatchData\\APA32\\ori_y_2000.npy",
                  "D:\\Study\\wlh\\adversarial_patch\\PatchData\\APA\\ori_y_2000.npy",
                  "D:\\Study\\wlh\\adversarial_patch\\PatchData\\APA64\\ori_y_2000.npy",
                  "D:\\Study\\wlh\\adversarial_patch\\PatchData\\GDPA\\ori_y_2000.npy",
                  "G:\\wlh\\chapters2\\AEDD_methods\\adv_data\\GDPA48\\ori_y_2000.npy",
                  "G:\\wlh\\chapters2\\AEDD_methods\\adv_data\\GDPA64\\ori_y_2000.npy"]
    adx_x_path = ["D:\\Study\\wlh\\adversarial_patch\\PatchData\\APA32\\APA_Vgg19_32patch_2000.npy",
                  "D:\\Study\\wlh\\adversarial_patch\\PatchData\\APA\\APA_Vgg19_43x43patch_2000.npy",
                  "D:\\Study\\wlh\\adversarial_patch\\PatchData\\APA64\\APA_Vgg19_64patch_2000.npy",
                  "D:\\Study\\wlh\\adversarial_patch\\PatchData\\GDPA\\GDPA_Vgg19_32x32patch_2000.npy",
                  "G:\\wlh\\chapters2\\AEDD_methods\\adv_data\\GDPA48\\GDPA_Vgg19_48patch_2000.npy",
                  "G:\\wlh\\chapters2\\AEDD_methods\\adv_data\\GDPA64\\GDPA_Vgg19_64patch_2000.npy"]
    mask_x_path = ["D:\\Study\\wlh\\adversarial_patch\\PatchData\\APA32\\APA_Vgg19_32mask_2000.npy",
                  "D:\\Study\\wlh\\adversarial_patch\\PatchData\\APA\\APA_Vgg19_43x43mask_2000.npy",
                  "D:\\Study\\wlh\\adversarial_patch\\PatchData\\APA64\\APA_Vgg19_64mask_2000.npy",
                  "D:\\Study\\wlh\\adversarial_patch\\PatchData\\GDPA\\GDPA_Vgg19_32x32mask_2000.npy",
                  "G:\\wlh\\chapters2\\AEDD_methods\\adv_data\\GDPA48\\GDPA_Vgg19_48mask_2000.npy",
                  "G:\\wlh\\chapters2\\AEDD_methods\\adv_data\\GDPA64\\GDPA_Vgg19_64mask_2000.npy"]
    thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]

    # # ---------------------------------外层循环加载数据，内层循环变更 threshold------------------------------
    for index in range(6):
        ori_y = np.load(ori_y_path[index])
        adv_x = np.load(adx_x_path[index])
        mask_x = np.load(mask_x_path[index])
        for threshold in thresholds:
            print("---------------------", ori_y_path[index].split('\\')[-2], "---", threshold, "---------------------------")
            # # # ----------------------------------计算掩码交并比---------------------------
            # ori_x = torch.tensor(ori_x)
            # ori_y = torch.tensor(ori_y)
            adv_x = torch.tensor(adv_x)
            mask_x = torch.tensor(mask_x)
            IoU = 0
            Coverage = 0
            for i in tqdm(range(400)):
                # ori_x_input = ori_x[i * 5:(i + 1) * 5].cuda()
                adv_x_input = adv_x[i * 5:(i + 1) * 5].cuda()
                # target = ori_y[i * 5:(i + 1) * 5]
                target_mask = mask_x[i * 5:(i + 1) * 5].cuda()

                pix2pix_adv = generator(adv_x_input)
                mask_predict = torch.abs(pix2pix_adv - adv_x_input)

                # adv_x_input = np.array(adv_x_input.cpu())
                # adv_x_input = np.transpose(adv_x_input, (0, 2, 3, 1))
                # mask_predict = np.array(mask_predict.detach().cpu())
                # mask_predict = np.transpose(mask_predict, (0, 2, 3, 1))
                # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
                # for i in range(2):
                #     for j in range(5):
                #         num = i * 5 + j + 1
                #         if i == 0:
                #             plt.subplot(2, 5, num)
                #             plt.imshow(adv_x_input[j])
                #             plt.axis('off')
                #         elif i == 1:
                #             plt.subplot(2, 5, num)
                #             plt.imshow(mask_predict[j])
                #             plt.axis('off')
                # plt.show()
            #
                IoU += comput_IoU(mask_predict, target_mask, threshold)
                Coverage += comput_Coverage(mask_predict, target_mask, threshold)
            #
            print("交并比：", IoU / 400)
            print("覆盖率：", Coverage / 400)

            # -------------------------------加载目标分类模型，利用得到的掩码覆盖对抗补丁，计算分类准确率---------------------------
            model = VGGNet_19()
            model.load_state_dict(torch.load(" "))
            model = model.cuda()
            model.eval()

            ori_y = torch.tensor(ori_y)
            adv_x = torch.tensor(adv_x)

            total = [0,0,0,0]
            correct = [0,0,0,0]
            acc_test = [0,0,0,0]
            for i in tqdm(range(400)):
                # ori_x_input = ori_x[i * 5:(i + 1) * 5].cuda()
                adv_x_input = adv_x[i * 5:(i + 1) * 5].cuda()
                target = ori_y[i * 5:(i + 1) * 5].cuda()

                pix2pix_adv = generator(adv_x_input)
                mask_predict = torch.abs(pix2pix_adv - adv_x_input)

                mask_predict = torch.where(mask_predict < threshold, 0, 1)

                hide = torch.ones((5,3,256,256)).cuda()
                mask_predict = torch.sum(mask_predict, dim=1)
                mask_predict = torch.where(mask_predict != 0, 1, 0)
                mask_predict = torch.unsqueeze(mask_predict, dim=1)
                hide_image = torch.where(mask_predict == 1, hide, adv_x_input)

                # 图像修复
                mask = np.array(torch.where(torch.sum(mask_predict, dim=1)!=0, 1, 0).cpu().detach())
                mask = mask.astype(np.uint8)
                image = np.array(hide_image.cpu()) * 255
                image = np.transpose(image, (0, 2, 3, 1))
                image = image.astype(np.uint8)
                adv_x_repair = []
                for i in range(5):
                    adv_x_repair.append(cv2.inpaint(image[i], mask[i], 3, cv2.INPAINT_TELEA))
                    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
                    # plt.imshow(adv_x_repair[i])
                    # plt.show()
                adv_x_repair = torch.tensor(np.transpose(adv_x_repair, (0, 3, 1, 2)), dtype=torch.float32).cuda()
                adv_x_repair = adv_x_repair/255

                # output = model(ori_x_input)
                # _, predicted = torch.max(output.data, dim=1)
                # total[0] += target.size(0)
                # correct[0] += (predicted == target).sum().item()

                # output = model(adv_x_input)
                # _, predicted = torch.max(output.data, dim=1)
                # total[1] += target.size(0)
                # correct[1] += (predicted == target).sum().item()

                output = model(hide_image)
                _, predicted = torch.max(output.data, dim=1)
                total[2] += target.size(0)
                correct[2] += (predicted == target).sum().item()

                output = model(adv_x_repair)
                _, predicted = torch.max(output.data, dim=1)
                total[3] += target.size(0)
                correct[3] += (predicted == target).sum().item()

            # acc_test[0] = correct[0] / total[0]
            # acc_test[1] = correct[1] / total[1]
            acc_test[2] = correct[2] / total[2]
            acc_test[3] = correct[3] / total[3]
            # print("干净样本准确率：%.5f" % acc_test[0])
            # print("对抗补丁准确率：%.5f" % acc_test[1])
            print("遮挡样本准确率：%.5f" % acc_test[2])
            print("修复样本准确率：%.5f" % acc_test[3])

    # # -----------------------------------可视化-------------------------------------
    # ori_x = torch.tensor(ori_x)
    # ori_y = torch.tensor(ori_y)
    # adv_x = torch.tensor(adv_x)
    # mask_x = torch.tensor(mask_x)
    # IoU = 0
    # for i in range(400):
    #     ori_x_input = ori_x[i * 5:(i + 1) * 5].cuda()
    #     adv_x_input = adv_x[i * 5:(i + 1) * 5].cuda()
    #     target = ori_y[i * 5:(i + 1) * 5]
    #     mask_x = mask_x[i * 5:(i + 1) * 5].cuda()
    #
    #     pix2pix_ori = generator(ori_x_input)
    #     pix2pix_adv = generator(adv_x_input)
    #
    #     real_A = np.array(ori_x_input.cpu())
    #     real_A = np.transpose(real_A, (0, 2, 3, 1))
    #
    #     fake_A = np.array(pix2pix_ori.detach().cpu())
    #     fake_A = np.transpose(fake_A, (0, 2, 3, 1))
    #
    #     real_B = np.array(adv_x_input.cpu())
    #     real_B = np.transpose(real_B, (0, 2, 3, 1))
    #
    #     fake_B = np.array(pix2pix_adv.detach().cpu())
    #     fake_B = np.transpose(fake_B, (0, 2, 3, 1))
    #
    #     plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    #     for i in range(6):
    #         for j in range(5):
    #             num = i * 5 + j + 1
    #             if i == 0:
    #                 plt.subplot(6, 5, num)
    #                 plt.imshow(real_A[j])
    #                 plt.axis('off')
    #             elif i == 1:
    #                 plt.subplot(6, 5, num)
    #                 plt.imshow(fake_A[j])
    #                 plt.axis('off')
    #             elif i == 2:
    #                 plt.subplot(6, 5, num)
    #                 plt.imshow(np.fabs(real_A[j] - fake_A[j]))
    #                 plt.axis('off')
    #             elif i == 3:
    #                 plt.subplot(6, 5, num)
    #                 plt.imshow(real_B[j])
    #                 plt.axis('off')
    #             elif i == 4:
    #                 plt.subplot(6, 5, num)
    #                 plt.imshow(fake_B[j])
    #                 plt.axis('off')
    #             elif i == 5:
    #                 plt.subplot(6, 5, num)
    #                 plt.imshow(np.fabs(real_B[j] - fake_B[j]))
    #                 plt.axis('off')
    #     plt.show()
