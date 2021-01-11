import os
import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from influence_pytorch import i_up_loss, i_pert_loss

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_1 = torch.nn.Linear(10, 100)
        self.layer_2 = torch.nn.Linear(100, 50)
        self.layer_3 = torch.nn.Linear(50, 1)
        self.relu = torch.nn.SELU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


def main():
    device = 'cuda:0'
    x, y = make_classification(n_samples=1000, n_features=10, n_classes=2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = Model()
    no_epochs = 10
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    for epoch in range(no_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model.forward(torch.from_numpy(x_train).float().to(device))
        loss = criterion(outputs, torch.from_numpy(y_train).float().to(device).view(-1, 1))
        loss.backward()
        optimizer.step()
        print('Epoch {}/{}: Training Loss: {:4f}'.format(epoch + 1, no_epochs, loss.item()))
    model.eval()
    eqn_2 = i_up_loss(x_train, y_train, x_test, y_test, model, model.layer_3.weight, device=device)
    eqn_5 = i_pert_loss(x_train, y_train, x_test, y_test, model, model.layer_3.weight, device=device)
    print(eqn_2)
    print(eqn_5)


if __name__ == '__main__':
    main()
