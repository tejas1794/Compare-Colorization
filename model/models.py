from torch import nn
import tensorflow as tf
from keras import applications
from keras.models import load_model
import os
import numpy as np
import cv2
import math



# DIRECTORY INFORMATION
OUT_DIR = os.path.join('static/')
# DATA INFORMATION
BATCH_SIZE = 1
# TRAINING INFORMATION
PRETRAINED = "my_model_colorization.h5"

class BaseColor(nn.Module):
    def __init__(self):
        super(BaseColor, self).__init__()

        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm


class CNN(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(CNN, self).__init__()

        model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model1 += [norm_layer(64), ]
        self.model1 = nn.Sequential(*model1)

        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model2 += [norm_layer(128), ]
        self.model2 = nn.Sequential(*model2)

        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model3 += [norm_layer(256), ]
        self.model3 = nn.Sequential(*model3)

        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model4 += [norm_layer(512), ]
        self.model4 = nn.Sequential(*model4)

        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ] + [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ] + [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ] + [nn.ReLU(True), ]
        model5 += [norm_layer(512), ]
        self.model5 = nn.Sequential(*model5)

        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ] + [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ] + [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ] + [nn.ReLU(True), ]
        model6 += [norm_layer(512), ]
        self.model6 = nn.Sequential(*model6)

        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model7 += [norm_layer(512), ]
        self.model7 = nn.Sequential(*model7)

        model8 = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ] + [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True), ]
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))


def load_cnn_trained_model(pretrained=True):
    """Loads model architecture, and if requested, loads pretrained weights from Author's website"""
    model = CNN()
    if pretrained:
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(
            model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',
                               map_location='cpu', check_hash=True))
    return model


class DATA():

    def __init__(self, imagepath):
        self.file = imagepath
        self.batch_size = BATCH_SIZE
        self.size = len(self.file)
        self.data_index = 0

    def read_img(self, filename):
        IMAGE_SIZE = 224
        MAX_SIDE = 1500
        img = cv2.imread(filename, 3)
        if img is None:
            print("Unable to read image: " + filename)
            return False, False, False, False, False
        height, width, channels = img.shape
        if height > MAX_SIDE or width > MAX_SIDE:
            print("Image " + filename + " is of size (" + str(height) + "," + str(width) + ").")
            print("The maximum image size allowed is (" + str(MAX_SIDE) + "," + str(MAX_SIDE) + ").")
            r = min(MAX_SIDE / height, MAX_SIDE / width)
            height = math.floor(r * height)
            width = math.floor(r * width)
            img = cv2.resize(img, (width, height))
            print("It has been resized to (" + str(height) + "," + str(width) + ")")
        labimg = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
        labimg_ori = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return True, np.reshape(labimg[:, :, 0], (IMAGE_SIZE, IMAGE_SIZE, 1)), labimg[:, :, 1:], img, np.reshape(
            labimg_ori[:, :, 0], (height, width, 1))

    def generate_batch(self):
        batch = []
        labels = []
        filelist = []
        labimg_oritList = []
        originalList = []
        for i in range(self.batch_size):
            filename = self.file
            ok, greyimg, colorimg, original, labimg_ori = self.read_img(filename)
            if ok:
                filelist.append(self.file)
                batch.append(greyimg)
                labels.append(colorimg)
                originalList.append(original)
                labimg_oritList.append(labimg_ori)
                self.data_index = (self.data_index + 1) % self.size
            break
        batch = np.asarray(batch) / 255  # values between 0 and 1
        labels = np.asarray(labels) / 255  # values between 0 and 1
        originalList = np.asarray(originalList)
        labimg_oritList = np.asarray(labimg_oritList) / 255
        return batch, labels, filelist, originalList, labimg_oritList


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)

    return result


def sample_images(imagepath):
    avg_ssim = 0
    avg_psnr = 0
    VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True)
    save_path = os.path.join(PRETRAINED)
    colorizationModel = load_model("model/" + save_path)
    test_data = DATA(imagepath)
    assert test_data.size >= 0, "Your list of images to colorize is empty. Please load images."
    total_batch = int(test_data.size / BATCH_SIZE)
    print("")
    print("number of images to colorize: " + str(test_data.size))
    print("total number of batches to colorize: " + str(total_batch))
    print("")
    if not os.path.exists(OUT_DIR):
        print('created save result path')
        os.makedirs(OUT_DIR)
    for b in range(total_batch):
        batchX, batchY, filelist, original, labimg_oritList = test_data.generate_batch()
        if batchX.any():
            predY, _ = colorizationModel.predict(np.tile(batchX, [1, 1, 1, 3]))
            predictVGG = VGG_modelF.predict(np.tile(batchX, [1, 1, 1, 3]))
            loss = colorizationModel.evaluate(np.tile(batchX, [1, 1, 1, 3]), [batchY, predictVGG], verbose=0)
            for i in range(BATCH_SIZE):
                originalResult = original[i]
                height, width, channels = originalResult.shape
                predY_2 = deprocess(predY[i])
                predY_2 = cv2.resize(predY_2, (width, height))
                labimg_oritList_2 = labimg_oritList[i]
                predResult_2 = reconstruct(deprocess(labimg_oritList_2), predY_2)
                ssim = tf.keras.backend.eval(tf.image.ssim(tf.convert_to_tensor(originalResult, dtype=tf.float32),
                                                           tf.convert_to_tensor(predResult_2, dtype=tf.float32),
                                                           max_val=255))
                psnr = tf.keras.backend.eval(tf.image.psnr(tf.convert_to_tensor(originalResult, dtype=tf.float32),
                                                           tf.convert_to_tensor(predResult_2, dtype=tf.float32),
                                                           max_val=255))
                avg_ssim += ssim
                avg_psnr += psnr
                save_path = os.path.join(OUT_DIR, "result2.jpg")
                cv2.imwrite(save_path, predResult_2)
                print("")
                print("Image " + str(i + 1) + "/" + str(BATCH_SIZE) + " in batch " + str(b + 1) + "/" + str(
                    total_batch) + ". From left to right: grayscale image to colorize, colorized image ( PSNR =",
                      "{:.8f}".format(psnr), ")")
                print("and ground truth image. Notice that PSNR has no sense in original black and white images.")
                print("")
                print("")
        break

    print("average ssim loss =", "{:.8f}".format(avg_ssim / (total_batch * BATCH_SIZE)))
    print("average psnr loss =", "{:.8f}".format(avg_psnr / (total_batch * BATCH_SIZE)))