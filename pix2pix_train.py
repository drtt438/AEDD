import argparse
import glob
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from torch.autograd import Variable

from ALD import GeneratorUNet, Discriminator, weights_init_normal
from target_models import VGGNet_19

# --------------------------------------设置参数-----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="change_light_and_add_noise", help="name of the dataset")
parser.add_argument("--loss_ratio", type=str, default="0.9-0.1", help="ratio of the loss function")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=1, help="epoch from which to start lr decay")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=1000, help="interval between sampling of images from generators"
)
opt = parser.parse_args()




class GtsrbDataset(Dataset):
    def __init__(self, root, transforms=None):
        imgs = []
        for i in range(43):
            root1 = root + str(i)
            allfile = glob.glob(root1 + "/*.png")
            for file in allfile:
                imgs.append((file,i))

        self.imgs = imgs
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(256),
                T.ToTensor(),
                # normalize
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index][0]
        label = self.imgs[index][1]

        data = Image.open(img_path)
        if data.mode != "RGB":
            data = data.convert("RGB")
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


# 在随机位置添加空白块
def mask_generation(image_size=(16, 3, 256, 256)):
    mask_length = int(random.choice([0, 32, 48, 64]))
    patch = np.ones((image_size[1], mask_length, mask_length))

    applied_patch = np.zeros(image_size)

    # patch location
    for i in range(image_size[0]):
        x_location = np.random.randint(low=0, high=image_size[2] - patch.shape[1])
        y_location = np.random.randint(low=0, high=image_size[3] - patch.shape[2])
        applied_patch[i, :, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch

    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask


def Train_pix2pix():
    global best_acc
    loss_D = 0
    for epoch in range(start_epoch, opt.n_epochs):
        generator.train()
        discriminator.train()
        pbar = tqdm(train_loader)
        print("start epoch {}".format(epoch))
        total = 0
        loss_C = 0
        loss_pix = 0
        for i, (image, target) in enumerate(pbar):
            # 在图像中添加随机空白块（0，32，48，64）
            # 将带有空白块的图像修复回原图
            # Model inputs
            applied_patch, mask = mask_generation(image_size=image.size())
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            masked_images = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) \
                                 + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            real_A = Variable(masked_images.type(Tensor))
            real_B = Variable(image.type(Tensor))
            label = Variable(target.type(Tensor))
            total += label.size(0)

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_B.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_B.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake1 = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake1, valid)

            # 全局像素损失
            loss_global_pixel = criterion_pixelwise(fake_B, real_B)
            # 局部像素损失
            fake_B_local = torch.mul((1 - mask.type(torch.FloatTensor)), fake_B.type(torch.FloatTensor))
            real_B_local = torch.mul((1 - mask.type(torch.FloatTensor)), real_B.type(torch.FloatTensor))
            loss_local_pixel = criterion_pixelwise(fake_B_local, real_B_local)
            loss_pixel = 0.5 * (loss_global_pixel + loss_local_pixel)

            # classify loss
            loss_classify = criterion_classify(model(fake_B), label.to(torch.int64))
            # Total loss  这里最好加个模型分类器的损失函数
            ratio = opt.loss_ratio.split("-")
            loss_G = float(ratio[0]) * (loss_GAN + lambda_pixel * loss_pixel) + float(ratio[1]) * loss_classify

            loss_C += loss_classify
            loss_pix += loss_pixel

            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator 20回合训练一次
            # ---------------------
            if i % 20 == 0:
                optimizer_D.zero_grad()

                # Real loss
                pred_real = discriminator(real_B, real_A)
                loss_real = criterion_GAN(pred_real, valid)

                # Fake loss
                pred_fake = discriminator(fake_B.detach(), real_A)
                loss_fake = criterion_GAN(pred_fake, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------
            # scheduler_G.step()

            pbar.set_postfix({"正在处理元素：": i, "G_loss：": loss_G.item(), "pixel_lose: ": loss_pix.item() / total,
                              "classify_loss: ": loss_C / total, "D_loss：": loss_D.item()})
        scheduler_G.step()

        # -----------------------------验证结果-----------------------------
        generator.eval()
        pbar = tqdm(val_loader)
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (image, target) in enumerate(pbar):
                # Model inputs
                real_A = Variable(image.type(Tensor))
                label = Variable(target.type(Tensor))

                output = model(generator(real_A))
                _, predicted = torch.max(output.data, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                pbar.set_postfix({"正在处理元素：": i, "acc：": correct / total})
        acc = correct / total
        checkpoint = {'model': generator.state_dict(), 'optimizer': optimizer_G.state_dict(), 'lr': scheduler_G.state_dict(),
                      'end_epoch': epoch, 'last_acc': acc, 'best_acc': best_acc}
        torch.save(checkpoint, "./save_models/generator_last.pth")
        torch.save(discriminator.state_dict(), "./save_models/discriminator_last.pth")

        if acc > best_acc:
            best_acc = acc
            checkpoint = {'model': generator.state_dict(), 'optimizer': optimizer_G.state_dict(),
                          'lr': scheduler_G.state_dict(),
                          'end_epoch': epoch, 'best_acc': best_acc}
            torch.save(checkpoint, "./save_models/generator_best.pth")
            print("save epoch {} model".format(epoch))


if __name__ == '__main__':
    # --------------目标分类网络-------------------
    model = VGGNet_19()
    model.load_state_dict(torch.load(" "))
    model = model.cuda()
    model.eval()

    # -------------------------------设置pix2pix图像修复网络网络结构、训练用的损失函数、加载目标防御模型-----------------------------
    cuda = True if torch.cuda.is_available() else False
    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_classify = torch.nn.CrossEntropyLoss()
    criterion_feature = torch.nn.MSELoss()

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator = Discriminator()

    if cuda:
        model = model.cuda()
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
        criterion_classify.cuda()
        criterion_feature.cuda()

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if opt.epoch != 0:
        # Load pretrained models
        checkpoint = torch.load("saved_models/%s/%s_diff_loss_generator_last.pth" % (opt.dataset_name, opt.loss_ratio))
        generator.load_state_dict(checkpoint['model'])
        discriminator.load_state_dict(
            torch.load("saved_models/%s/%s_diff_loss_discriminator_last.pth" % (opt.dataset_name, opt.loss_ratio)))
        start_epoch = checkpoint['end_epoch'] + 1
        best_acc = checkpoint['best_acc']
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0000125, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0000125, betas=(opt.b1, opt.b2))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        start_epoch = 0
        best_acc = 0
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Optimizers
    scheduler_G = StepLR(optimizer_G, step_size=opt.decay_epoch, gamma=0.2)
    model.eval()

    # -------------------------------------------加载训练数据-------------------------------------------------
    train_dataset = GtsrbDataset(root=" ")
    train_size = int(0.7 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset_split, val_dataset_split = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset_split, batch_size=opt.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset_split, batch_size=opt.batch_size, shuffle=True)
    print(len(train_loader))
    print(len(val_loader))

    Train_pix2pix()

