import torch
import torch.nn as nn


class myconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(myconv,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convfor = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.convfor(x)
        return x 
    
class unetpp(nn.Module):
    def __init__(self, num_classes, deep_supervision = False):
        super(unetpp, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filter = [32, 64, 128, 256, 512] 
        
        self.conv0_0 = myconv(3, self.filter[0])   
        self.conv1_0 = myconv(self.filter[0], self.filter[1])
        self.conv2_0 = myconv(self.filter[1], self.filter[2])
        self.conv3_0 = myconv(self.filter[2], self.filter[3])
        self.conv4_0 = myconv(self.filter[3], self.filter[4])
        
        self.conv0_1 = myconv(self.filter[0]*2, self.filter[0])
        self.conv0_2 = myconv(self.filter[0]*3, self.filter[0])
        self.conv0_3 = myconv(self.filter[0]*4, self.filter[0])
        self.conv0_4 = myconv(self.filter[0]*5, self.filter[0])
        
        self.conv1_1 = myconv(self.filter[1]*2, self.filter[1])
        self.conv1_2 = myconv(self.filter[1]*3, self.filter[1])
        self.conv1_3 = myconv(self.filter[1]*4, self.filter[1])
        
        self.conv2_1 = myconv(self.filter[2]*2, self.filter[2])
        self.conv2_2 = myconv(self.filter[2]*3, self.filter[2])
        
        self.conv3_1 = myconv(self.filter[3]*2, self.filter[3])
        
        self.pool = nn.MaxPool2d(2)
        
        self.up3_1 = nn.ConvTranspose2d(self.filter[4], self.filter[3], kernel_size=4, stride=2, padding=1)
        
        self.up2_1 = nn.ConvTranspose2d(self.filter[3], self.filter[2], kernel_size=4, stride=2, padding=1)
        self.up2_2 = nn.ConvTranspose2d(self.filter[3], self.filter[2], kernel_size=4, stride=2, padding=1)
        
        self.up1_1 = nn.ConvTranspose2d(self.filter[2], self.filter[1], kernel_size=4, stride=2, padding=1)
        self.up1_2 = nn.ConvTranspose2d(self.filter[2], self.filter[1], kernel_size=4, stride=2, padding=1)
        self.up1_3 = nn.ConvTranspose2d(self.filter[2], self.filter[1], kernel_size=4, stride=2, padding=1)
        
        self.up0_1 = nn.ConvTranspose2d(self.filter[1], self.filter[0], kernel_size=4, stride=2, padding=1)
        self.up0_2 = nn.ConvTranspose2d(self.filter[1], self.filter[0], kernel_size=4, stride=2, padding=1)
        self.up0_3 = nn.ConvTranspose2d(self.filter[1], self.filter[0], kernel_size=4, stride=2, padding=1)
        self.up0_4 = nn.ConvTranspose2d(self.filter[1], self.filter[0], kernel_size=4, stride=2, padding=1)
        
        self.loss_head0_1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(self.filter[0], self.num_classes, 3, padding = 1),
        )
        self.loss_head0_2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(self.filter[0], self.num_classes, 3, padding = 1),
        )
        self.loss_head0_3 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(self.filter[0], self.num_classes, 3, padding = 1),
        )
        self.loss_head0_4 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(self.filter[0], self.num_classes, 3, padding = 1),
        )
    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        x0_1 = torch.cat([self.up0_1(x1_0), x0_0], 1)
        x0_1 = self.conv0_1(x0_1)
        
        x1_1 = torch.cat([self.up1_1(x2_0), x1_0], 1)
        x1_1 = self.conv1_1(x1_1)
        
        x2_1 = torch.cat([self.up2_1(x3_0), x2_0], 1)
        x2_1 = self.conv2_1(x2_1)
        
        x3_1 = torch.cat([self.up3_1(x4_0), x3_0], 1)
        x3_1 = self.conv3_1(x3_1)
        
        x0_2 = torch.cat([self.up0_2(x1_1), x0_1, x0_0], 1)
        x0_2 = self.conv0_2(x0_2)
        
        x1_2 = torch.cat([self.up1_2(x2_1), x1_1, x1_0], 1)
        x1_2 = self.conv1_2(x1_2)
        
        x2_2 = torch.cat([self.up2_2(x3_1), x2_1, x2_0], 1)
        x2_2 = self.conv2_2(x2_2)
        
        x0_3 = torch.cat([self.up0_3(x1_2), x0_2, x0_1, x0_0], 1)
        x0_3 = self.conv0_3(x0_3)
        
        x1_3 = torch.cat([self.up1_3(x2_2), x1_2, x1_1, x1_0], 1)
        x1_3 = self.conv1_3(x1_3)
        
        x0_4 = torch.cat([self.up0_4(x1_3), x0_3, x0_2, x0_1, x0_0], 1)
        x0_4 = self.conv0_4(x0_4)
        
        if self.deep_supervision:
            output1 = self.loss_head0_1(x0_1)
            output2 = self.loss_head0_2(x0_2)
            output3 = self.loss_head0_3(x0_3)
            output4 = self.loss_head0_4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.loss_head0_4(x0_4)
        
        
if __name__ == "__main__":
    print("deep_supervision: False")
    deep_supervision = False
    device = torch.device('cpu')
    inputs = torch.randn((1, 3, 96, 96)).to(device)
    model = unetpp(num_classes=2, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    print(outputs.shape)    
    
    print("deep_supervision: True")
    deep_supervision = True
    model = unetpp(num_classes=2, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    for out in outputs:
     print(out.shape)
   
        
        