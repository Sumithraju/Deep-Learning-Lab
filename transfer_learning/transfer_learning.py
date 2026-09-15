import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os

from PIL import Image
from tempfile import TemporaryDirectory
cudnn.benchmark = True
plt.ion()
# DATA TRANSFORMS
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
}
# LOAD DATASET
data_dir = 'hymenoptera_data'

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        data_transforms[x]
    )
    for x in ['train', 'val']
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    for x in ['train', 'val']
}

dataset_sizes = {
    x: len(image_datasets[x])
    for x in ['train', 'val']
}


class_names = image_datasets['train'].classes

# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Classes:", class_names)

# DISPLAY IMAGE
def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # reverse normalization
    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)


# Get one validation batch

inputs, classes = next(iter(dataloaders['val']))

out = torchvision.utils.make_grid(inputs)

imshow(
    out,
    title=[class_names[x] for x in classes]
)
# TRAINING FUNCTION
def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25
):

    since = time.time()

    with TemporaryDirectory() as tempdir:

        best_model_params_path = os.path.join(
            tempdir,
            'best_model_params.pth'
        )

        # Save initial model
        torch.save(
            model.state_dict(),
            best_model_params_path
        )

        best_acc = 0.0


        for epoch in range(num_epochs):

            print(
                'Epoch {}/{}'.format(
                    epoch + 1,
                    num_epochs
                )
            )

            print('-' * 10)

            print(
                'Learning rate:',
                optimizer.param_groups[0]['lr']
            )


            # Each epoch has training and validation
            for phase in ['train', 'val']:

                if phase == 'train':
                    model.train()

                else:
                    model.eval()


                running_loss = 0.0
                running_corrects = 0


                # Iterate through batches
                for inputs, labels in dataloaders[phase]:

                    inputs = inputs.to(device)
                    labels = labels.to(device)


                    # Clear previous gradients
                    optimizer.zero_grad()


                    # Enable gradient only during training
                    with torch.set_grad_enabled(
                        phase == 'train'
                    ):

                        # Forward pass
                        outputs = model(inputs)

                        _, preds = torch.max(
                            outputs,
                            1
                        )

                        loss = criterion(
                            outputs,
                            labels
                        )


                        # Backpropagation
                        if phase == 'train':

                            loss.backward()

                            optimizer.step()


                    # Statistics
                    running_loss += (
                        loss.item()
                        * inputs.size(0)
                    )

                    running_corrects += torch.sum(
                        preds == labels
                    )


                # Change learning rate after training phase
                if phase == 'train':
                    scheduler.step()


                # Epoch loss
                epoch_loss = (
                    running_loss
                    / dataset_sizes[phase]
                )


                # Epoch accuracy
                epoch_acc = (
                    running_corrects.double()
                    / dataset_sizes[phase]
                )


                print(
                    f'{phase} '
                    f'Loss: {epoch_loss:.4f} '
                    f'Acc: {epoch_acc:.4f}'
                )


                # Save best validation model
                if (
                    phase == 'val'
                    and epoch_acc > best_acc
                ):

                    best_acc = epoch_acc

                    torch.save(
                        model.state_dict(),
                        best_model_params_path
                    )


        # Training finished
        time_elapsed = time.time() - since


        print(
            'Training complete in '
            '{:.0f}m {:.0f}s'.format(
                time_elapsed // 60,
                time_elapsed % 60
            )
        )


        print(
            'Best val Acc: {:4f}'.format(
                best_acc
            )
        )


        # Load best model weights
        model.load_state_dict(
            torch.load(
                best_model_params_path,
                map_location=device,
                weights_only=True
            )
        )


    return model

# VISUALIZE MODEL PREDICTIONS

def visualize_model(model, num_images=6):

    was_training = model.training

    model.eval()

    images_so_far = 0

    fig = plt.figure()


    with torch.no_grad():

        for i, (inputs, labels) in enumerate(
            dataloaders['val']
        ):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(
                outputs,
                1
            )


            for j in range(inputs.size(0)):

                images_so_far += 1


                ax = plt.subplot(
                    num_images // 2,
                    2,
                    images_so_far
                )

                ax.axis('off')

                ax.set_title(
                    'Predicted: {}'.format(
                        class_names[preds[j]]
                    )
                )


                imshow(
                    inputs.cpu().data[j]
                )


                if images_so_far == num_images:

                    model.train(
                        mode=was_training
                    )

                    return


        model.train(
            mode=was_training
        )

# METHOD 1 — FINE-TUNING
model_ft = models.resnet18(
    weights='IMAGENET1K_V1'
)


# Number of input features to final FC layer
num_ftrs = model_ft.fc.in_features


# Replace ImageNet 1000-class layer with 2-class layer
model_ft.fc = nn.Linear(
    num_ftrs,
    len(class_names)
)


model_ft = model_ft.to(device)


# Loss function
criterion = nn.CrossEntropyLoss()


# Optimize ALL model parameters
optimizer_ft = optim.SGD(
    model_ft.parameters(),
    lr=0.001,
    momentum=0.9
)


# Reduce LR by factor 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_ft,
    step_size=7,
    gamma=0.1
)


# Train
model_ft = train_model(
    model_ft,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    num_epochs=25
)
# Visualize predictions
visualize_model(model_ft)
# METHOD 2 — FIXED FEATURE EXTRACTOR
model_conv = models.resnet18(
    weights='IMAGENET1K_V1'
)
# Freeze pretrained layers
for param in model_conv.parameters():

    param.requires_grad = False
# Replace final FC layer
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(
    num_ftrs,
    len(class_names)
)
model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()
# IMPORTANT:
# Train ONLY the newly created FC layer
optimizer_conv = optim.SGD(
    model_conv.fc.parameters(),
    lr=0.001,
    momentum=0.9
)
exp_lr_scheduler_conv = lr_scheduler.StepLR(
    optimizer_conv,
    step_size=7,
    gamma=0.1
)
model_conv = train_model(
    model_conv,
    criterion,
    optimizer_conv,
    exp_lr_scheduler_conv,
    num_epochs=25
)
visualize_model(model_conv)
# PREDICT A SINGLE IMAGE
def visualize_model_predictions(
    model,
    img_path
):
    was_training = model.training

    model.eval()


    img = Image.open(
        img_path
    ).convert('RGB')
    img = data_transforms['val'](
        img
    )
    # Add batch dimension
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():

        outputs = model(img)

        _, preds = torch.max(
            outputs,
            1
        )
        ax = plt.subplot()
        ax.axis('off')
        ax.set_title(
            'Predicted: {}'.format(
                class_names[preds[0]]
            )
        )
        imshow(
            img.cpu().data[0]
        )
    model.train(
        mode=was_training
    )
# Change this path if required
visualize_model_predictions(
    model_conv,
    img_path='hymenoptera_data/val/bees/72100438_73de9f17af.jpg'
)
plt.ioff()
plt.show()