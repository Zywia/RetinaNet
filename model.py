from torch import nn
import torchvision


class FPN(nn.Module):
    def __init__(self, feature_size=256):
        super(FPN, self).__init__()

        C3_size = 512
        C4_size = 1024
        C5_size = 2048


        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        # return out.contiguous().view(out.shape[0], -1, 4)
        return out


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, 3, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        # out1 = out.permute(0, 2, 3, 1)

        # batch_size, width, height, channels = out1.shape

        # out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        # return out2.contiguous().view(x.shape[0], -1, self.num_classes)
        # print (out.permute(0, 2, 3, 1).shape)
        return out.permute(0, 2, 3, 1).contiguous().view(-1, 3)



class RetinaNet(nn.Module):
    def __init__(self):
        super(RetinaNet, self).__init__()

        mod = torchvision.models.resnet50(pretrained=True, progress=False)
        self.C3 = nn.Sequential(mod.conv1, mod.bn1, mod.relu, mod.maxpool, mod.layer1, mod.layer2)

        self.C4 = mod.layer3
        self.C5 = mod.layer4

        for x in [self.C3, self.C4, self.C5]:
            for y in x.parameters():
                y.requires_grad = False

        self.fpn = FPN()
        # self.regresion_P3 = RegressionModel(256)
        # self.regresion_P4 = RegressionModel(256)
        # self.regresion_P5 = RegressionModel(256)
        # self.regresion_P6 = RegressionModel(256)
        # self.regresion_P7 = RegressionModel(256)

        self.classifier_P3 = ClassificationModel(256)
        self.classifier_P4 = ClassificationModel(256)
        self.classifier_P5 = ClassificationModel(256)
        self.classifier_P6 = ClassificationModel(256)
        self.classifier_P7 = ClassificationModel(256)

    def forward(self, images):
        C3 = self.C3(images)
        C4 = self.C4(C3)
        C5 = self.C5(C4)

        P3_x, P4_x, P5_x, P6_x, P7_x = self.fpn([C3, C4, C5])

        # return [[ self.classifier_P3(P3_x), self.regresion_P3(P3_x)],\
        #         [self.classifier_P4(P4_x), self.regresion_P4(P4_x)],\
        #         [self.classifier_P5(P5_x), self.regresion_P5(P5_x)],\
        #         [self.classifier_P6(P6_x), self.regresion_P6(P6_x)],\
        #         [self.classifier_P7(P7_x), self.regresion_P7(P7_x)] ]
        return [self.classifier_P3(P3_x), self.classifier_P4(P4_x),
                self.classifier_P5(P5_x), self.classifier_P6(P6_x),
                self.classifier_P7(P7_x)]

