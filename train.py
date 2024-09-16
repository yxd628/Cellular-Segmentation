import numpy as np
import torch.optim.lr_scheduler
from d2l import torch as d2l
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from dataset import CustomDataset
from torch.utils.data import DataLoader,random_split
from model import unetpp
from torchvision import transforms

batch_size = 4
num_classes = 2
learning_rate = 0.01
epochs_num = 3

#load data
# Define transformations, can include normalization, resizing, etc.
transform = transforms.Compose([
    transforms.ToTensor() # Resize images if needed
])
dataset = CustomDataset(image_dir='/content/drive/MyDrive/553project/Data_augmented-20240416T234324Z-001/Data_augmented/TissueImages_color_normalized_augmented', mask_dir='/content/drive/MyDrive/553project/Data_augmented-20240416T234324Z-001/Data_augmented/GroundTruth_augmented', transform=transform)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # Remaining 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

#model(net) to put it on cuda
#model = unetpp(num_classes=2).cuda()
model = unetpp(num_classes=2)
#load pretrained model
#model.load_state_dict(torch.load(r"checkpoints/Unet++_25.pth"),strict=False)

# Use adam optimizer to train
opti = optim.SGD(model.parameters(), lr=0.1)
# Learning rate scheduler
sche = torch.optim.lr_scheduler.StepLR(opti, step_size=50, gamma=0.1, last_epoch=-1)



# loss function: Cross Entropy Loss
lossf = nn.CrossEntropyLoss(ignore_index=255)

def calculate_iou(pred, target):
     # Assuming the prediction and target are probabilities that need to be thresholded to create binary masks
    pred = pred > 0.5  # Threshold predictions to get binary mask
    target = target > 0.5  # Ensure target is also a binary mask

    # Convert to integer if not already
    pred = pred.int()
    target = target.int()

    intersection = (pred & target).sum((1, 2))  # Compute intersection
    union = (pred | target).sum((1, 2))  # Compute union

    iou = (intersection + 1e-6) / (union + 1e-6)  # Calculate IoU, add small epsilon to avoid division by zero
    return iou.mean()  # Mean IoU over batch

def evaluate_accuracy(net, dataloader, device):
    """Evaluate accuracy of a segmentation model."""
    net.eval()  # Set the model to evaluation mode
    total_iou = 0
    num_batches = 0

    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            # Assuming output of the network is already a binary mask or has been thresholded
            pred = torch.argmax(output, dim=1)  # Convert to binary output if needed
            iou = calculate_iou(pred, target)
            total_iou += iou
            num_batches += 1

    return (total_iou / num_batches).item()

def train(model, train_iter, test_iter, loss, trainer, epochs_num, scheduler, devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    if torch.cuda.device_count() > 0:
        devices = [i for i in range(torch.cuda.device_count())]
        net = nn.DataParallel(model, device_ids=devices).to('cuda:' + str(devices[0]))
    else:
        net = model.to('cpu')
        print("No CUDA devices found, using CPU instead.")
   

    loss_list = []
    train_acc_list = []
    test_acc_list = []
    epochs_list = []
    time_list = []
    for epoch in range(epochs_num):
        # Sum of training loss, sum of training accuracy, # of examples, and # of predictions
        # These 4 quantities are represented as accumulator[0-3] respectively
        accumulator = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels.long().squeeze(1), loss, trainer, devices)
            accumulator.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            
        test_acc = evaluate_accuracy(net, test_iter, 'cuda:' + str(devices[0]))
     
        scheduler.step()
        print(f"epoch {epoch+1} --- loss {accumulator[0] / accumulator[2]:.3f} ---  train acc {accumulator[1] / accumulator[3]:.3f} --- test acc {test_acc:.3f} --- cost time {timer.sum()}")

        # save data
        df = pd.DataFrame()
        loss_list.append(accumulator[0] / accumulator[2])
        train_acc_list.append(accumulator[1] / accumulator[3])
        test_acc_list.append(test_acc)
        epochs_list.append(epoch)
        time_list.append(timer.sum())

        df['epoch'] = epochs_list
        df['loss'] = loss_list
        df['train_acc'] = train_acc_list
        df['test_acc'] = test_acc_list
        df['time'] = time_list
        df.to_excel("Unet++.xlsx")
        # 保存模型
        if np.mod(epoch + 1, 5) == 0:
            torch.save(model.state_dict(), f'checkpoints/Unet++_{epoch + 1}.pth')

train(model, train_loader, test_loader, lossf, opti, epochs_num, sche)