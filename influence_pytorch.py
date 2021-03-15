import torch
import random
import numpy as np
from torch.autograd import grad


def hessian_vector_product(ys, xs, v):
    """
    Compute Hessian Vector Product
    Ensure all functions of the model are twice differentiable
    i.e. change ReLUs to SELUs
    :param ys: train loss, [tensor, with grad]
    :param xs: parameters [tensor, with grad]
    :param v: gradient of test loss, [tensor with grad]
    :return: H * gradient of test loss
    """
    J = grad(ys, xs, create_graph=True)[0]
    grads = grad(J, xs, v, retain_graph=True)
    del J, ys, v
    torch.cuda.empty_cache()
    return grads


def lissa(train_loss, test_loss, layer_weight, model):
    """
    Estimate Inverse Hessian Vector Product
    :param train_loss: train loss (some type of cross-entropy loss), [tensor, with grad]
    :param test_loss: test loss (some type of cross-entopy loss), [tensor, with grad]
    :param layer_weight: weights of the last layer of the model e.g. model.layer_2.weight - suggested by Koh, Pang 2017
    :param model: pytorch model, [pytorch model object]
    :return: H^-1 * gradient of test loss
    """
    scale = 10
    damping = 0.01
    num_samples = 1
    v = grad(test_loss, layer_weight)[0]
    cur_estimate = v.clone()
    prev_norm = 1
    diff = prev_norm
    count = 0
    while diff > 0.00001 and count < 10000:
        try:
            hvp = hessian_vector_product(train_loss, layer_weight, cur_estimate)
            cur_estimate = [a + (1 - damping) * b - c / scale for (a, b, c) in zip(v, cur_estimate, hvp)]
            cur_estimate = torch.squeeze(torch.stack(cur_estimate)).view(1, -1)
            model.zero_grad()
            numpy_est = cur_estimate.detach().cpu().numpy()
            numpy_est = numpy_est.reshape(1, -1)

            if (count % 100 == 0):
                print("Recursion at depth %s: norm is %.8lf" % (count, np.linalg.norm(np.concatenate(numpy_est))))
            count += 1
            diff = abs(np.linalg.norm(np.concatenate(numpy_est)) - prev_norm)
            prev_norm = np.linalg.norm(np.concatenate(numpy_est))
            ihvp = [b / scale for b in cur_estimate]
            ihvp = torch.squeeze(torch.stack(ihvp))
            ihvp = [a / num_samples for a in ihvp]
            ihvp = torch.squeeze(torch.stack(ihvp))
        except Exception:
            print('LiSSA Failed')
            return np.zeros_like(v.detach().cpu().numpy())
    return ihvp.detach()


def i_up_params(x_train, y_train, model, layer_weight, n=1, std=0.2,
                criterion=torch.nn.BCEWithLogitsLoss(), device='cpu'):
    """
    Estimate the effect that upweighting a training instance will have on the parameters of the model
    :param x_train: training data, [n_samples, n_feats]
    :param y_train: training labels, [n_samples]
    :param model: PyTorch model object
    :param layer_weight: Weight of layer to compute influence at. Koh, Pang 2017 suggest final layer
    :param n: If you'd like to smooth influence by using an average of the neighborhood increase n, [int]
              This intuition is from Smilkov, et.al. 2017 (SmoothGrad)
    :param std: Standard deviation of noise to add to sample during smoothing
    :param criterion: Some type of Cross-entropy function
    :param device: Device to compute on. Much faster on cuda devices
    :return: list len training set: H^-1 * gradient of training instance loss with respect to last layer in model
    """
    eqn_1 = []
    x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    for itr in range(n):
        if n > 1:
            np.random.seed(random.randint(0, 10000000))
            noise = np.random.normal(0, std, x_train.size())
            x_train = x_train.cpu() + noise  # add noise to test data
        if 'cuda' in device:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            model = model.to(device)
        train_loss = criterion(model(x_train), y_train.view(-1, 1))
        for i in range(len(x_train)):
            x = x_train[i]
            x.requires_grad = True
            model.zero_grad()
            x_out = model(x)
            x_loss = criterion(x_out, y_train[i].view(-1))
            ihvp = lissa(train_loss, x_loss, layer_weight, model)
            eqn_1.append(-ihvp.detach().cpu().numpy())
    return eqn_1


def i_up_loss(x_train, y_train, x_test, y_test, model, layer_weight, n=1, std=0.2,
              criterion=torch.nn.BCEWithLogitsLoss(), device='cpu'):
    """
    Estimate the effect that upweighting a training instance will have on the loss of a test instance
    :param x_train: training data, [n_samples, n_feats]
    :param y_train: training labels, [n_samples]
    :param x_test: test instance or test data, [n_samples, n_feats]
    :param y_test: test labels, [n_samples]
    :param model: PyTorch model object
    :param layer_weight: Weight of layer to compute influence at. Koh, Pang 2017 suggest final layer
    :param n: If you'd like to smooth influence by using an average of the neighborhood increase n, [int]
              This intuition is from Smilkov, et.al. 2017 (SmoothGrad)
    :param std: Standard deviation of noise to add to sample during smoothing
    :param criterion: Some type of Cross-entropy function
    :param device: Device to compute on. Much faster on cuda devices
    :return: gradient of training loss of each traning instance * H^-1 * gradient of test loss
    """
    eqn_2 = []
    x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()
    for itr in range(n):
        if n > 1:
            np.random.seed(random.randint(0, 10000000))
            noise = np.random.normal(0, std, x_test.size())
            x_test = x_test.cpu() + noise  # add noise to test data
        if 'cuda' in device:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            model = model.to(device)
        train_loss = criterion(model(x_train), y_train.view(-1, 1))
        test_loss = criterion(model(x_test), y_test.view(-1, 1))

        ihvp = lissa(train_loss, test_loss, layer_weight, model)

        for i in range(len(x_train)):
            x = x_train[i]
            x.requires_grad = True
            model.zero_grad()
            x_out = model(x)
            x_loss = criterion(x_out, y_train[i].view(-1))
            grads = grad(x_loss, layer_weight, create_graph=True)[0]
            grads = grads.squeeze()
            grads = grads.view(1, -1).squeeze()

            infl = (torch.dot(ihvp.view(1, -1).squeeze(), grads.view(-1, 1).squeeze()) / len(x_train))
            eqn_2.append(-infl.detach().cpu().numpy())
    return eqn_2


def i_pert_loss(x_train, y_train, x_test, y_test, model, layer_weight, n=1, std=0.2,
                criterion=torch.nn.BCEWithLogitsLoss(), device='cpu'):
    """
    Estimate the effect that upweighting the inputs of a training instance will have on the loss of a test instance
    :param x_train: training data, [n_samples, n_feats]
    :param y_train: training labels, [n_samples]
    :param x_test: test instance or test data, [n_samples, n_feats]
    :param y_test: test labels, [n_samples]
    :param model: PyTorch model object
    :param layer_weight: Weight of layer to compute influence at. Koh, Pang 2017 suggest final layer
    :param n: If you'd like to smooth influence by using an average of the neighborhood increase n, [int]
              This intuition is from Smilkov, et.al. 2017 (SmoothGrad)
    :param std: Standard deviation of noise to add to sample during smoothing
    :param criterion: Some type of Cross-entropy function
    :param device: Device to compute on. Much faster on cuda devices
    :return: gradient with respect to input (gradient of training loss of each traning instance * H^-1 * gradient of test loss)
    """
    eqn_5 = []
    x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()
    for itr in range(n):
        if n > 1:
            np.random.seed(random.randint(0, 10000000))
            noise = np.random.normal(0, std, x_test.size())
            x_test = x_test.cpu() + noise  # add noise to test data
        if 'cuda' in device:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            model = model.to(device)
        train_loss = criterion(model(x_train), y_train.view(-1, 1))
        test_loss = criterion(model(x_test), y_test.view(-1, 1))

        ihvp = lissa(train_loss, test_loss, layer_weight, model)

        x = x_train
        x.requires_grad = True
        x_out = model(x)
        x_loss = criterion(x_out, y_train.view(-1, 1))
        grads = grad(x_loss, layer_weight, create_graph=True)[0]
        grads = grads.squeeze()
        grads = grads.view(1, -1).squeeze()
        infl = (torch.dot(ihvp.view(-1, 1).squeeze(), grads)) / len(x_train)
        i_pert = grad(infl, x, retain_graph=False)
        i_pert = i_pert[0]

        eqn_5.append(np.sum(-i_pert.detach().cpu().numpy(), axis=0))
        model.zero_grad()

    return np.asarray(eqn_5).squeeze()
