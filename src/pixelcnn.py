import torch
from torch import nn, optim
from torch.functional import F
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from torch.distributions.beta import Beta

from . import  datasets


def empty_image(device, image_size=28):
    return torch.zeros((1, 1, image_size, image_size)).to(device)


def generate_image(model, device, image=None, skip=0, k=50.0):
    model.eval()

    if image is None: image = empty_image(device)

    with torch.no_grad():
        for i in range(skip, image.size().numel()):
            x, y = divmod(i, image.size(-1))
            out = model(image)
            prob = torch.sigmoid(out[0, 0, x, y])
            image[0, 0, x, y] = Beta(prob*k, (1-prob)*k).sample()
    return image.cpu().numpy()[0, 0]







class OccludedConv2d(torch.nn.Conv2d):
    def __init__(self, *args, include_center=False, **kwargs):
        super().__init__(*args, **kwargs)

        w,h = self.weight.size()[-2:]
        mask = torch.zeros(w*h)
        mask[:w * h // 2 + int(include_center)] = 1
        mask = mask.view(w, h)

        self.register_buffer('mask', mask)

    def forward(self, x) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class Model(nn.Module):
    def __init__(self, input_channels=1, n_filters=64, kernel_size=7, n_layers=7):
        super().__init__()
        # The first layer uses mask type 'A'
        self.input_conv = OccludedConv2d(input_channels, n_filters, kernel_size, include_center=False, padding=kernel_size//2)

        self.hidden_convs = nn.ModuleList([
            OccludedConv2d(n_filters, n_filters, kernel_size, include_center=True, padding=kernel_size//2)
            for _ in range(n_layers)
        ])

        self.output_conv = nn.Conv2d(n_filters, input_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.input_conv(x))
        for conv in self.hidden_convs:
            x = F.relu(conv(x))
        x = self.output_conv(x)
        return x





def trainer(dataset, model, optimizer, compute_loss, epochs=10):

    progress = tqdm(total=len(dataset)*epochs)
    for e in range(epochs):
        for input, _ in dataset:
            model.train()
            optimizer.zero_grad()

            output = model(input)
            loss = compute_loss(output, input)
            loss.backward()
            optimizer.step()

            progress.update()



def do_train(dataset):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = datasets.mnist(device)

    train_by_digit = [
        Subset(train, [i for i, l in enumerate(train) if l[1] == d])
        for d in range(10)
    ]

    model = Model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    compute_loss = nn.BCEWithLogitsLoss()

    dataset = DataLoader(dataset, batch_size=16, shuffle=True)
    trainer(dataset, model, optimizer,  compute_loss, epochs=200)
