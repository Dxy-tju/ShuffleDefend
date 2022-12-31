from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

def fgsm_whitebox( model,
                   X,
                   y,
                   epsilon=8.0/255.0):
    X_fgsm = Variable(X.data, requires_grad=True)

    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()

    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_fgsm), y)
    loss.backward()

    X_fgsm = Variable(torch.clamp(X_fgsm.data + epsilon * X_fgsm.grad.data.sign(), 0.0, 1.0), requires_grad=True)
    return X_fgsm

def ifgsm(model, 
          X, 
          y, 
          epsilon=8.0/255.0, 
          alpha=2.0/255.0, 
          num_iter=10):
    for i in range(0, num_iter):
      x_adv = fgsm_whitebox(model, X, y, alpha)
      x_adv = torch.clamp(X-epsilon, X+epsilon)
    return x_adv

def jacobian(predictions, x, nb_classes):
    list_derivatives = []

    for class_ind in range(nb_classes):
        outputs = predictions[:, class_ind]
        derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True)
        list_derivatives.append(derivatives)

    return list_derivatives

class DeepFool(object):
    def __init__(self, nb_candidate=10, overshoot=0.02, max_iter=50, clip_min=0.0, clip_max=1.0):
        self.nb_candidate = nb_candidate
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.clip_min = clip_min
        self.clip_max = clip_max

    def attack(self, model, x):
        device = x.device

        with torch.no_grad():
            logits = model(x)
        self.nb_classes = logits.size(-1)
        assert self.nb_candidate <= self.nb_classes, 'nb_candidate should not be greater than nb_classes'

        # preds = logits.topk(self.nb_candidate)[0]
        # grads = torch.stack(jacobian(preds, x, self.nb_candidate), dim=1)
        # grads will be the shape [batch_size, nb_candidate, image_size]

        adv_x = x.clone().requires_grad_()

        iteration = 0
        logits = model(adv_x)
        current = logits.argmax(dim=1)
        if current.size() == ():
            current = torch.tensor([current])
        w = torch.squeeze(torch.zeros(x.size()[1:])).to(device)
        r_tot = torch.zeros(x.size()).to(device)
        original = current

        while ((current == original).any and iteration < self.max_iter):
            predictions_val = logits.topk(self.nb_candidate)[0]
            gradients = torch.stack(jacobian(predictions_val, adv_x, self.nb_candidate), dim=1)
            with torch.no_grad():
                for idx in range(x.size(0)):
                    pert = float('inf')
                    if current[idx] != original[idx]:
                        continue
                    for k in range(1, self.nb_candidate):
                        w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                        f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                        pert_k = (f_k.abs() + 0.00001) / w_k.view(-1).norm()
                        if pert_k < pert:
                            pert = pert_k
                            w = w_k

                    r_i = pert * w / w.view(-1).norm()
                    r_tot[idx, ...] = r_tot[idx, ...] + r_i

            adv_x = torch.clamp(r_tot + x, self.clip_min, self.clip_max).requires_grad_()
            logits = model(adv_x)
            current = logits.argmax(dim=1)
            if current.size() == ():
                current = torch.tensor([current])
            iteration = iteration + 1

        adv_x = torch.clamp((1 + self.overshoot) * r_tot + x, self.clip_min, self.clip_max)
        return adv_x

def pgd_whitebox(model,
                  X,
                  y,
                  epsilon=8.0/255.0,
                  num_steps=10,
                  step_size=2.0/255.0):
    X_pgd = Variable(X.data, requires_grad=True)
    if True:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return X_pgd