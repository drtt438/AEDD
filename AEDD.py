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

from ALD import GeneratorUNet
from target_models import VGGNet_19


def comput_IoU(mask_predict, mask_x):
    mask_predict = torch.sum(mask_predict, dim=1)
    mask_predict = torch.where(mask_predict < 0.5, 0, 1)
    mask_x = torch.sum(mask_x, dim=1)
    mask_x = torch.where(mask_x != 0, 1, 0)

    iou = 0
    for i in range(5):
        result = mask_x[i] + mask_predict[i]
        Intersection = (result == 2).sum().item()
        Union = (result != 0).sum().item()
        iou += Intersection / Union
    return iou / 5


def comput_Coverage(mask_predict, mask_x):
    mask_predict = torch.sum(mask_predict, dim=1)
    mask_predict = torch.where(mask_predict < 0.5, 0, 1)
    mask_x = torch.sum(mask_x, dim=1)
    mask_x = torch.where(mask_x != 0, 1, 0)

    Coverage = 0
    for i in range(5):
        original = (mask_x[i] == 1).sum().item()
        result = mask_x[i] + mask_predict[i]
        cover = (result == 2).sum().item()
        Coverage += cover / original
    return Coverage / 5


def batch_mask_detection(images):
    mask = torch.zeros((5,3,256,256))
    for i in range(len(images)):
        image = images[i]
        # 用来进行轮廓标记的图像
        imgContour = image.copy()
        # 灰度转换
        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 模糊、边缘提取
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        imgCanny = cv2.Canny(imgBlur, 50, 50)
        # 膨胀
        kernel = np.ones((2, 2), np.uint8)
        binary_dilation = cv2.dilate(imgCanny, kernel)
        # binary_dilation = cv2.dilate(binary_dilation, kernel)
        # 轮廓提取                                                   只提取外轮廓       保存物体边界上所有连续的轮廓点
        contours, hierarchy = cv2.findContours(binary_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 对轮廓进行遍历，取出想要的对抗补丁的轮廓
        for cnt in contours:
            # 计算轮廓包围的面积
            area = cv2.contourArea(cnt)
            # 考虑当面积在什么范围时处理
            if area > 650 and area < 4500:
            # if area < 4500:
                # 画轮廓线(蓝色)(后面几个参数都是什么意思？？)
                cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
                # 将光滑的轮廓线折线化
                peri = cv2.arcLength(cnt, True)
                # 近似折线                     折线距离轮廓线的近似距离， 折线是否封闭
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                # 将检测区域置1
                x, y, w, h = cv2.boundingRect(approx)
                mask[i, :, y:y+h, x:x+w] = 1

    return mask


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
    adv_x_path = ["D:\\Study\\wlh\\adversarial_patch\\PatchData\\APA32\\APA_Vgg19_32patch_2000.npy",
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

    # ----------------------------------加载目标分类模型，利用得到的掩码覆盖对抗补丁，计算分类准确率---------------------------
    model = VGGNet_19()
    model.load_state_dict(torch.load(" "))
    model = model.cuda()
    model.eval()

    for index in range(6):
        ori_y = np.load(ori_y_path[index])
        adv_x = np.load(adv_x_path[index])
        # mask_x = np.load(mask_x_path[index])

        # ori_x = torch.tensor(ori_x)
        ori_y = torch.tensor(ori_y)
        adv_x = torch.tensor(adv_x)

        total = [0,0,0,0]
        correct = [0,0,0,0]
        acc_test = [0,0,0,0]
        for i in tqdm(range(400)):
            # ori_x_input = ori_x[i * 5:(i + 1) * 5].cuda()
            adv_x_input = adv_x[i * 5:(i + 1) * 5].cuda()
            target = ori_y[i * 5:(i + 1) * 5].cuda()

            # -------------------------------------ALD模块--------------------------------
            pix2pix_adv = generator(adv_x_input)
            mask_predict = torch.abs(pix2pix_adv - adv_x_input)

            # -------------------------------------EDD模块--------------------------------
            mask_predict = np.array(mask_predict.cpu().detach()) * 255
            mask_predict = np.transpose(mask_predict, (0, 2, 3, 1))
            mask_predict = mask_predict.astype(np.uint8)
            mask_predict = batch_mask_detection(mask_predict)
            mask_predict = torch.tensor(mask_predict)
            mask_predict = mask_predict.cuda()

            hide = torch.ones((5,3,256,256)).cuda()
            adv_x_hide = torch.where(mask_predict == 1, hide, adv_x_input)
            # -------------------------------------图像修复模块--------------------------------
            mask = np.array(torch.where(torch.sum(mask_predict, dim=1)!=0, 1, 0).cpu().detach())
            mask = mask.astype(np.uint8)

            image = np.array(adv_x_hide.cpu().detach()) * 255
            image = np.transpose(image, (0, 2, 3, 1))
            image = image.astype(np.uint8)
            adv_x_repair = []
            for i in range(5):
                adv_x_repair.append(cv2.inpaint(image[i], mask[i], 3, cv2.INPAINT_TELEA))
            adv_x_repair = torch.tensor(np.transpose(adv_x_repair, (0, 3, 1, 2)), dtype=torch.float32).cuda()
            adv_x_repair = adv_x_repair/255

            # output = model(ori_x_input)
            # _, predicted = torch.max(output.data, dim=1)
            # total[0] += target.size(0)
            # correct[0] += (predicted == target).sum().item()

            output = model(adv_x_input)
            _, predicted = torch.max(output.data, dim=1)
            total[1] += target.size(0)
            correct[1] += (predicted == target).sum().item()

            output = model(adv_x_hide)
            _, predicted = torch.max(output.data, dim=1)
            total[2] += target.size(0)
            correct[2] += (predicted == target).sum().item()

            output = model(adv_x_repair)
            _, predicted = torch.max(output.data, dim=1)
            total[3] += target.size(0)
            correct[3] += (predicted == target).sum().item()

        # acc_test[0] = correct[0] / total[0]
        acc_test[1] = correct[1] / total[1]
        acc_test[2] = correct[2] / total[2]
        acc_test[3] = correct[3] / total[3]
        # print("干净样本准确率：%.5f" % acc_test[0])
        print("对抗补丁准确率：%.5f" % acc_test[1])
        print("遮挡样本准确率：%.5f" % acc_test[2])
        print("修复样本准确率：%.5f" % acc_test[3])

    # # -----------------------------------可视化-------------------------------------
    # ori_x = torch.tensor(ori_x)
    # ori_y = torch.tensor(ori_y)
    # adv_x = torch.tensor(adv_x)
    # # mask_x = torch.tensor(mask_x)
    # IoU = 0
    # for i in range(400):
    #     ori_x_input = ori_x[i * 5:(i + 1) * 5].cuda()
    #     adv_x_input = adv_x[i * 5:(i + 1) * 5].cuda()
    #     target = ori_y[i * 5:(i + 1) * 5]
    #     # mask_x = mask_x[i * 5:(i + 1) * 5].cuda()
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
