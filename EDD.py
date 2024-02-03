import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from target_models import VGGNet_19


def comput_IoU(mask_predict, mask_x):
    mask_predict = torch.sum(mask_predict, dim=1)
    mask_predict = torch.where(mask_predict < 1, 0, 1)
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
    mask_predict = torch.where(mask_predict < 1, 0, 1)
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

        # 轮廓提取                                                   只提取外轮廓       保存物体边界上所有连续的轮廓点
        contours, hierarchy = cv2.findContours(binary_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 对轮廓进行遍历，取出想要的对抗补丁的轮廓
        for cnt in contours:
            # 计算轮廓包围的面积
            area = cv2.contourArea(cnt)

            # 考虑当面积在什么范围时处理
            if area > 650 and area < 4500:
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

    # # ---------------------------------外层循环加载数据------------------------------
    for index in range(6):
        ori_y = np.load(ori_y_path[index])
        adv_x = np.load(adx_x_path[index])
        mask_x = np.load(mask_x_path[index])
        print("---------------------", ori_y_path[index].split('\\')[-2], "---", "---------------------------")

        # # ----------------------------------计算掩码交并比---------------------------
        mask_x = torch.tensor(mask_x)
        IoU = 0
        Coverage = 0
        for i in tqdm(range(400)):
            adv_x_input = adv_x[i * 5:(i + 1) * 5]
            target_mask = mask_x[i * 5:(i + 1) * 5].cuda()

            # 这里输入需要cv2格式，即需要转换成（batch,w,h,c）的形式
            adv_input = adv_x_input * 255
            adv_input = np.transpose(adv_input, (0, 2, 3, 1))
            adv_input = adv_input.astype(np.uint8)
            mask_predict = batch_mask_detection(adv_input)
        # ----------------- 预测掩码可视化 ------------
        #     # adv_x_input = np.array(adv_x_input)
        #     # adv_x_input = np.transpose(adv_x_input, (0, 2, 3, 1))
        #     # target_mask = np.array(target_mask.cpu())
        #     # target_mask = np.transpose(target_mask, (0, 2, 3, 1))
        #     # mask_predict = np.array(mask_predict.cpu())
        #     # mask_predict = np.transpose(mask_predict, (0, 2, 3, 1))
        #     #
        #     # # 这里输入需要PIL图像格式，即需要转换成RGB格式（其实这里都是0-1掩码，转不转无所谓）
        #     # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
        #     # for i in range(3):
        #     #     for j in range(5):
        #     #         num = i * 5 + j + 1
        #     #         if i == 0:
        #     #             plt.subplot(3, 5, num)
        #     #             plt.imshow(target_mask[j])
        #     #             plt.axis('off')
        #     #         elif i == 1:
        #     #             plt.subplot(3, 5, num)
        #     #             plt.imshow(mask_predict[j])
        #     #             plt.axis('off')
        #     #         elif i == 2:
        #     #             plt.subplot(3, 5, num)
        #     #             plt.imshow(adv_x_input[j])
        #     #             plt.axis('off')
        #     # plt.show()
            mask_predict = torch.tensor(mask_predict)
            mask_predict = mask_predict.cuda()
            IoU += comput_IoU(mask_predict, target_mask)
            Coverage += comput_Coverage(mask_predict, target_mask)
        #
        print("交并比：", IoU / 400)
        print("覆盖率：", Coverage / 400)

        # --------------------------加载目标分类模型，利用得到的掩码覆盖对抗补丁，计算分类准确率---------------------------
        model = VGGNet_19()
        model.load_state_dict(torch.load(" "))
        model = model.cuda()
        model.eval()

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

            # 这里输入需要cv2格式，即需要转换成BGR格式和（batch,w,h,c）的形式
            adv_input = np.array(adv_x_input.cpu()) * 255
            adv_input = np.transpose(adv_input, (0, 2, 3, 1))
            adv_input = adv_input.astype(np.uint8)
            mask_predict = batch_mask_detection(adv_input)
            mask_predict = torch.tensor(mask_predict)
            mask_predict = mask_predict.cuda()

            hide = torch.ones((5,3,256,256)).cuda()
            adv_x_hide = torch.where(mask_predict == 1, hide, adv_x_input)

            # ------------- 图像修复 ----------------
            image = np.array(adv_x_hide.cpu()) * 255
            image = np.transpose(image, (0, 2, 3, 1))
            image = image.astype(np.uint8)
            mask = np.array(torch.where(torch.sum(mask_predict, dim=1) != 0, 1, 0).cpu().detach())
            mask = mask.astype(np.uint8)
            adv_x_repair = []
            for i in range(5):
                adv_x_repair.append(cv2.inpaint(image[i], mask[i], 3, cv2.INPAINT_TELEA))
                # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
                # plt.imshow(adv_x_repair[i])
                # plt.show()
            adv_x_repair = torch.tensor(np.transpose(adv_x_repair, (0, 3, 1, 2)), dtype=torch.float32).cuda()
            adv_x_repair = adv_x_repair / 255

            # output = model(ori_x_input)
            # _, predicted = torch.max(output.data, dim=1)
            # total[0] += target.size(0)
            # correct[0] += (predicted == target).sum().item()

            # output = model(adv_x_input)
            # _, predicted = torch.max(output.data, dim=1)
            # total[1] += target.size(0)
            # correct[1] += (predicted == target).sum().item()

            output = model(adv_x_hide)
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
