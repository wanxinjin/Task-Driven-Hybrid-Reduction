# implementation of gradient descent
import numpy as np


class Vanilla_GD:

    def __init__(self):
        self.learning_rate = 0
        self.epoch_k = 0

    def step(self, current_para, para_grad):
        self.epoch_k += 1
        return current_para - self.learning_rate * para_grad


# implementation of Momentum gradient descent
class Momentum_GD:

    def __init__(self):
        self.learning_rate = 0.0
        self.momentum = 0.9
        self.epoch_k = 0
        self.previous_para_grad = 0.0

    def step(self, current_para, para_grad):
        update = self.momentum * self.previous_para_grad + self.learning_rate * para_grad
        para_next = current_para - update
        self.epoch_k += 1
        self.previous_para_grad = update
        return para_next


# implementation of Nesterov gradient descent
class Nesterov_GD:

    def __init__(self):
        self.learning_rate = 0.0
        self.momentum = 0.9
        self.epoch_k = 0
        self.previous_update = 0.0

    def project(self, current_para):
        return current_para - self.momentum * self.previous_update

    def step(self, current_para, para_grad):
        update = self.momentum * self.previous_update + self.learning_rate * para_grad
        para_next = current_para - update
        self.epoch_k += 1
        self.previous_update = update
        return para_next


# implementation of AdaGrad gradient descent
class AdaGrad:

    def __init__(self):
        self.learning_rate = None
        self.epsilon = 1e-8
        self.epoch_k = 0
        self.grad_sum = 0.0

    def step(self, current_para, para_grad):
        self.grad_sum += para_grad ** 2
        update = para_grad / np.sqrt(self.grad_sum + self.epsilon)
        para_next = current_para - self.learning_rate * update
        self.epoch_k += 1
        return para_next


# implementation of AdaDelta gradient descent
class AdaDelta:

    def __init__(self):
        self.epsilon = 1e-8
        self.gamma = 0.5
        self.square_grad = 0.0
        self.square_update = 0.0
        self.epoch_k = 0

    def step(self, current_para, para_grad):
        self.square_grad = (self.gamma * self.square_grad) + ((1. - self.gamma) * (para_grad ** 2))
        update = -(np.sqrt(self.square_update + self.epsilon)) / (np.sqrt(self.square_grad + self.epsilon)) * para_grad
        self.square_update = (self.gamma * self.square_update) + ((1. - self.gamma) * (update ** 2))
        para_next = current_para + update
        self.epoch_k += 1
        return para_next


# implementation of Adam gradient descent
class Adam:

    def __init__(self, learning_rate=1e-2, decay=None, epsilon=1e-6, grad_clip=None):
        self.beta1 = 0.9
        self.beta2 = 0.9
        self.learning_rate = learning_rate
        self.mt = 0.0
        self.vt = 0.0
        self.epoch_k = 0
        self.epsilon = epsilon
        self.counter = 0
        self.decay = decay
        self.grad_clip = grad_clip

    def step(self, current_param, param_grad):

        # clip the grad if enabled
        if self.grad_clip is not None:
            param_grad = np.clip(param_grad, -self.grad_clip, self.grad_clip)

        # update the first momentum
        self.mt = self.beta1 * self.mt + (1.0 - self.beta1) * param_grad
        # update the second momentum
        self.vt = self.beta2 * self.vt + (1.0 - self.beta2) * param_grad ** 2
        # compute  bias-corrected estimate of mt
        mt_hat = self.mt / (1.0 - self.beta1 ** (self.epoch_k + 1))
        vt_hat = self.vt / (1.0 - self.beta2 ** (self.epoch_k + 1))

        # update the current parameter
        para_next = current_param - (self.learning_rate / (self.epsilon + np.sqrt(vt_hat))) * mt_hat
        self.epoch_k += 1

        # update the internal state
        self.counter = self.counter + 1

        # decay the learning rate if enabled
        if self.decay is not None:
            self.learning_rate = self.learning_rate * self.decay

        return para_next
