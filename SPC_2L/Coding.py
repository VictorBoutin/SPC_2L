import torch
import torch.nn.functional as f
import numpy as np


class ML_FISTA(object):

    def __init__(self, network, LossFunction, etas_0=None, alpha=1.1, th=5e-4, max_iter=1000, accell=True, mode=None, init=None):

        self.params = {'etas_0': etas_0,
                       'alpha': alpha,
                       #'do_line_search': do_line_search,
                       'mode':mode,
                       'max_inter': max_iter,
                       'th': th,
                       'accell': True}

        self.network = network
        self.LF = LossFunction

        if etas_0 is not None:
            self.etas_0 = etas_0
        else:
            self.etas_0 = [1 for _ in range(self.network.nb_layers)]

        self.alpha = alpha
        self.etas = self.etas_0
        self.max_iter = max_iter
        self.th = th
        self.layers = self.network.layers
        #self.do_line_search = do_line_search
        self.mode = mode ## could be 'eigen' or 'line_search'
        self.accell = accell
        self.init = init

    def line_search(self, X, gamma_z, grad, Loss, i, max_iter=100):

        condition = False
        gamma_saved = gamma_z[i]

        k = 0

        while (not condition):

            k += 1

            gamma_z[i] = self.LF.prox_G(gamma_saved, grad, self.etas[i], i)

            F = self.LF.F(X, gamma_z, i)
            G = self.LF.G(gamma_z[i], i)

            L = F + G

            Q = Loss + ((gamma_z[i] - gamma_saved) * grad).sum() + (gamma_z[i] - gamma_saved).pow(2).sum().div(2 * self.etas[i]) + G

            if (L <= Q).all() or (k > max_iter):
                condition = True
            else:
                self.etas[i] = self.etas[i] / (self.alpha ** k)

        return gamma_z[i]

    def coding(self, X, flag=None, gamma_in=None, softmax=False, labels=None):

        gamma = [None] * self.network.nb_layers
        gamma_old = [None] * self.network.nb_layers
        gamma_z = [None] * self.network.nb_layers
        Loss = [None] * self.network.nb_layers
        if gamma_in is None:
            gamma_in = [None] * self.network.nb_layers

        if flag is None:
            flag = [True] * self.network.nb_layers

        for i in range(self.network.nb_layers):
            if gamma_in[i] is not None:
                gamma[i] = gamma_in[i]
            else:
                if i > 0:
                    if self.init=='Conv':
                        gamma[i] = self.layers[i].forward(self.layers[i-1].to_next(gamma[i-1]))
                    else :
                        gamma[i] = torch.zeros_like(self.layers[i].forward(self.layers[i-1].to_next(gamma[i-1])))
                else:
                    if self.init=='Conv':
                        gamma[i] = self.layers[i].forward(X)
                    else :
                        gamma[i] = torch.zeros_like(self.layers[i].forward(X))


            gamma_old[i] = gamma[i]
            gamma_z[i] = gamma[i]

        #if self.do_line_search:
        if self.mode == 'line_search':
            self.etas = [self.etas_0[i] for i in range(self.network.nb_layers)]
        elif self.mode == 'eigen':
            self.etas = [1 / get_eigenvalue(gamma[i].size(), self.layers[i]) for i in range(self.network.nb_layers)]



        t_old = 1
        it = 0

        while it < self.max_iter:

            t_new = (1 + np.sqrt(1 + 4 * (t_old ** 2))) / 2

            for i in range(self.network.nb_layers):

                if flag[i]:


                    gamma_z[i].requires_grad = True

                    Loss[i] = self.LF.F(X, gamma_z, i, labels=labels)
                    Loss[i].backward()

                    gamma_z[i].requires_grad = False

                    grad = gamma_z[i].grad
                    if self.mode == 'line_search':
                        gamma[i] = self.line_search(X, gamma_z, grad, Loss[i], i)
                    if softmax == True and i == (self.network.nb_layers -1):
                        gamma[i] = self.LF.Soft(gamma_z[i], grad, self.etas[i], i)
                    else :
                        gamma[i] = self.LF.prox_G(gamma_z[i], grad, self.etas[i], i)

                    gamma_z[i] = gamma[i] + (gamma[i] - gamma_old[i]).mul((t_old - 1) / t_new)
                    gamma_old[i] = gamma[i]

                else:

                    Loss[i] = self.LF.F(X, gamma_z, i, labels=labels)
                    gamma_z[i] = gamma[i]

            if it == 0:
                Loss_old = [Loss[i] for i in range(self.network.nb_layers)]

            delta = [(Loss[i] - Loss_old[i]).detach().abs() / Loss_old[i] for i in range(self.network.nb_layers)]

            condition = True
            for i in range(self.network.nb_layers):
                condition = condition and (delta[i] < self.th).all()

            if it > 3 and condition:
                break

            Loss_old = [Loss[i] for i in range(self.network.nb_layers)]

            if self.accell:
                t_old = t_new

            it += 1

        return gamma_z, it, Loss, delta

class ML_Lasso(object):

    def __init__(self, network, lambdas, norm = None):

        self.network = network
        self.layers = network.layers
        self.lambdas = lambdas
        self.norm = norm
        if self.norm is None:
            self.norm = [1 for i in range(self.network.nb_layers+1)]

    def F(self, X, gamma, i, do_feedback=True, labels=None, softmax=False):


        if softmax and i==self.network.nb_layers-1:
            pred = self.layers[i].backward(f.softmax(gamma[i],  dim=1))
        else :
            pred = self.layers[i].backward(gamma[i])

        if i == 0:
            Loss = (X - pred).pow(2).sum().mul(self.layers[i].a /2).div(self.norm[i])
        else:
            Loss = (self.layers[i - 1].to_next(gamma[i - 1]) - pred).pow(2).sum().mul(self.layers[i].a / 2).div(self.norm[i])



        if i < (self.network.nb_layers - 1) and do_feedback:
            if softmax and i == self.network.nb_layers - 2:
                feedback = self.layers[i + 1].backward(f.softmax(gamma[i + 1],dim=1))
            else :
                feedback = self.layers[i + 1].backward(gamma[i + 1])
            Loss += (self.layers[i].to_next(gamma[i]) - feedback).pow(2).div(self.layers[i].v ** 2).sum().mul(
                self.layers[i].b / 2).div(self.norm[i+1])

        if (i == self.network.nb_layers-1) and (labels is not None):
            Loss += (f.softmax(self.layers[i].to_next(gamma[i]), dim=1) - labels).pow(2).div(
                self.layers[i].v ** 2).sum().mul(self.layers[i].b / 2)

        return Loss

    def F_d(self, X, gamma, i, label=None):

        pred = self.layers[i].backward(gamma[i])

        if i == 0:
            Loss = (X - pred).pow(2).sum().mul(self.layers[i].a /2).div(self.norm[i])
        else:
            Loss = (self.layers[i - 1].to_next(gamma[i - 1]) - pred).pow(2).sum().mul(self.layers[i].a / 2).div(self.norm[i])

        if i == self.network.nb_layers-1 and (label is not None):
            Loss += f.cross_entropy(self.layers[i-1].forward(gamma[i-2]),label)


        return Loss

    def F_v(self, X, gamma, i, kwargs_next={}, kwargs_previous={}):

        pred = self.layers[i].backward(self.layers[i].v * gamma[i])

        if i == 0:
            Loss = (X - pred).pow(2).sum().mul(self.layers[i].a / 2)
        else:
            Loss = (self.layers[i - 1].to_next(gamma[i - 1]) - pred).pow(2).sum().mul(self.layers[i].a / 2)

        gamma[i] = gamma[i].detach()
        return Loss

    def G(self, gamma, i):
        return self.lambdas[i]*gamma.abs().sum()

    def prox_G(self, gamma, grad, eta, i):
        return f.relu(gamma - eta * grad - eta * self.lambdas[i])

    def Soft(self, gamma, grad, eta, i):
        return f.softmax(gamma - eta*grad, dim=1)
'''
class ML_Lasso_sup(object):

    def __init__(self, network, lambdas, norm = None):

        self.network = network
        self.layers = network.layers
        self.lambdas = lambdas
        self.norm = norm
        if self.norm is None:
            self.norm = [1 for i in range(self.network.nb_layers+1)]

    def F(self, X, gamma, i, do_feedback=True, labels=None, softmax=False):


        if softmax and i==self.network.nb_layers-1:
            pred = self.layers[i].backward(f.softmax(gamma[i],  dim=1))
        else :
            #pred = f.tanh(self.layers[i].backward(gamma[i]))
            pred = self.layers[i].backward(gamma[i])

        if i == 0:
            Loss = (X - pred).pow(2).sum().mul(self.layers[i].a /2).div(self.norm[i])
        else:
            Loss = (self.layers[i - 1].to_next(gamma[i - 1]) - pred).pow(2).sum().mul(self.layers[i].a / 2).div(self.norm[i])



        if i < (self.network.nb_layers - 1) and do_feedback:
            if softmax and i == self.network.nb_layers - 2:
                feedback = self.layers[i + 1].backward(f.softmax(gamma[i + 1],dim=1))
            else :
                #feedback = f.tanh(self.layers[i + 1].backward(gamma[i + 1]))
                feedback = self.layers[i + 1].backward(gamma[i + 1])
            Loss += (self.layers[i].to_next(gamma[i]) - feedback).pow(2).div(self.layers[i].v ** 2).sum().mul(
                self.layers[i].b / 2).div(self.norm[i+1])

        if (i == self.network.nb_layers-1) and (labels is not None):
            Loss += (f.softmax(self.layers[i].to_next(gamma[i]), dim=1) - labels).pow(2).div(
                self.layers[i].v ** 2).sum().mul(self.layers[i].b / 2)

            #Loss += (self.layers[i].to_next(gamma[i]) - labels).pow(2).div(
            #    self.layers[i].v ** 2).sum().mul(self.layers[i].b / 2)
        return Loss

    def F_v(self, X, gamma, i, kwargs_next={}, kwargs_previous={}):

        #if self.layers[i].transform is not None:
        #    gamma[i] = self.layers[i].to_next(gamma[i], **kwargs_next)
        #    gamma[i] = self.layers[i].to_previous(gamma[i], **kwargs_previous)
        #    pred = self.layers[i].backward(gamma[i])

        #else:
        pred = self.layers[i].backward(self.layers[i].v * gamma[i])

        if i == 0:
            Loss = (X - pred).pow(2).sum().mul(self.layers[i].a / 2)
        else:
            Loss = (self.layers[i - 1].to_next(gamma[i - 1]) - pred).pow(2).sum().mul(self.layers[i].a / 2)

        gamma[i] = gamma[i].detach()
        return Loss

    def G(self, gamma, i):
        return self.lambdas[i]*gamma.abs().sum()

    def prox_G(self, gamma, grad, eta, i):
        return f.relu(gamma - eta * grad - eta * self.lambdas[i])

    def Soft(self, gamma, grad, eta, i):
        return f.softmax(gamma - eta*grad, dim=1)


class ML_Lasso_Hybrid(object):

    def __init__(self, network, lambdas, idx_ITML, norm=None):

        self.network = network
        self.layers = network.layers
        self.lambdas = lambdas
        self.idx_ITML = idx_ITML
        self.norm = norm
        if self.norm is None:
            self.norm=[1 for i in range(self.network.nb_layers)]

    def F(self, X, gamma, i, do_feedback=True, labels=None, softmax=False):


        if softmax and i==self.network.nb_layers-1:
            pred = self.layers[i].backward(f.softmax(gamma[i],  dim=1))
        else :
            pred = self.layers[i].backward(gamma[i])

        if i == 0:
            Loss = (X - pred).pow(2).sum().mul(self.layers[i].a /2)
        else:
            Loss = (self.layers[i - 1].to_next(gamma[i - 1]) - pred).pow(2).sum().mul(self.layers[i].a / 2)



        if i < (self.network.nb_layers - 1) and do_feedback:
            if softmax and i == self.network.nb_layers - 2:
                feedback = self.layers[i + 1].backward(f.softmax(gamma[i + 1],dim=1))
            else:
                feedback = self.layers[i + 1].backward(gamma[i + 1])
            Loss += (self.layers[i].to_next(gamma[i]) - feedback).pow(2).div(self.layers[i].v ** 2).sum().mul(self.layers[i].b / 2)


        if (i == self.network.nb_layers-1) and (labels is not None):
            Loss += (f.softmax(self.layers[i].to_next(gamma[i]), dim=1) - labels).pow(2).div(
                self.layers[i].v ** 2).sum().mul(self.layers[i].b / 2)

            #Loss += (self.layers[i].to_next(gamma[i]) - labels).pow(2).div(
            #    self.layers[i].v ** 2).sum().mul(self.layers[i].b / 2)
        return Loss

    def F_v(self, X, gamma, i, kwargs_next={}, kwargs_previous={}):

        #if self.layers[i].transform is not None:
        #    gamma[i] = self.layers[i].to_next(gamma[i], **kwargs_next)
        #    gamma[i] = self.layers[i].to_previous(gamma[i], **kwargs_previous)
        #    pred = self.layers[i].backward(gamma[i])

        #else:
        pred = self.layers[i].backward(self.layers[i].v * gamma[i])

        if i == 0:
            Loss = (X - pred).pow(2).sum().mul(self.layers[i].a / 2)
        else:
            Loss = (self.layers[i - 1].to_next(gamma[i - 1]) - pred).pow(2).sum().mul(self.layers[i].a / 2)

        gamma[i] = gamma[i].detach()
        return Loss

    def G(self, gamma, i):
        return self.lambdas[i]*gamma.abs().sum()

    def prox_G(self, gamma, grad, eta, i):
        if (self.idx_ITML is not None) and (i in self.idx_ITML):
            to_return = hard_ITML(gamma - eta * grad, self.lambdas[i])
        else:
            to_return = f.relu(gamma - eta * grad - eta * self.lambdas[i])
        return to_return

'''
'''
class ML_Hybrid_Lasso(object):

    def __init__(self, network, lambdas, idx_ITML=None):

        self.network = network
        self.layers = network.layers
        self.lambdas = lambdas
        self.idx_ITML=idx_ITML

    def F(self, X, gamma, i, feedback=True):

        pred = self.layers[i].backward(gamma[i])

        if i == 0:
            Loss = (X - pred).pow(2).sum().mul(self.layers[i].a /2)
        else:
            Loss = (self.layers[i - 1].to_next(gamma[i - 1]) - pred).pow(2).sum().mul(self.layers[i].a / 2)

        if i < (self.network.nb_layers - 1) and feedback:
            feedback = self.layers[i + 1].backward(gamma[i + 1])
            Loss += (self.layers[i].to_next(gamma[i]) - feedback).pow(2).sum().mul(self.layers[i].b / (2 * self.layers[i].v ** 2))

        return Loss

    def G(self, gamma, i):
        return self.lambdas[i]*gamma.abs().sum()

    def prox_G(self, gamma, grad, eta, i):
        if (self.idx_ITML is not None) and (i in self.idx_ITML):
            to_return = hard_ITML(gamma - eta * grad, self.lambdas[i])
        else:
            to_return = f.relu(gamma - eta * grad - eta * self.lambdas[i])
        return to_return


class ML_GroupLasso(object):
    def __init__(self, network, lambdas, omegas=None, groups=None, mu=1e-9):

        self.network = network
        self.layers = network.layers
        self.lambdas = lambdas
        self.omegas = omegas

        if groups is not None:
            self.groups = groups
        else:
            self.groups = [None] * self.network.nb_layers

        self.mu = mu

    def F(self, X, gamma, i, do_feedback=True):

        pred = self.layers[i].backward(gamma[i])

        if i == 0:
            Loss = (X - pred).pow(2).sum().mul(self.layers[i].a / 2)
        else:
            Loss = (self.layers[i - 1].to_next(gamma[i - 1]) - pred).pow(2).sum().mul(self.layers[i].a / 2)

        if do_feedback:
            if i < (self.network.nb_layers - 1):
                feedback = self.layers[i + 1].backward(gamma[i + 1])
                Loss += (self.layers[i].to_next(gamma[i]) - feedback).pow(2).div(self.layers[i].v ** 2).sum().mul(
                    self.layers[i].b / 2)

            if self.groups[i] is not None:
                Loss += self.omegas[i] * l1_approx(self.groups[i](gamma[i]), mu=self.mu).sum()

        return Loss

    def G(self, gamma, i):
        return self.lambdas[i] * gamma.abs().sum()

    def prox_G(self, gamma, grad, eta, i):
        return f.relu(gamma - eta * grad - eta * self.lambdas[i])
'''

def HS(x, th):
    return f.relu(x - th) - f.relu(-x - th) + (f.relu(x - th) - f.relu(-x - th)).sign() * th


def l1_approx(x, mu=1e-8):
    mask = (HS(x.detach().sqrt(), mu).abs()).sign()
    return mask * ((x + (1 - mask)).sqrt() - (mu / 2)) + \
           (1 - mask) * x.div(mu * 2)
'''

class ML_ITML(object):

    def __init__(self, network, ks):

        self.network = network
        self.layers = network.layers
        self.ks = ks

    def F(self, X, gamma, i, do_feedback=True):

        pred = self.layers[i].backward(gamma[i])

        if i == 0:
            Loss = (X - pred).pow(2).sum().div(2)
        else:
            Loss = (self.layers[i - 1].to_next(gamma[i - 1]) - pred).pow(2).sum().mul(self.layers[i].a / 2)

        if i < (self.network.nb_layers - 1) and do_feedback:
            feedback = self.layers[i + 1].backward(gamma[i + 1])
            Loss += (self.layers[i].to_next(gamma[i]) - feedback).pow(2).sum().mul(self.layers[i].b / (2 * self.layers[i].v ** 2))

        return Loss

    #def G(self, gamma, i):
    #   return self.lambdas[i]*gamma.abs().sum()

    def prox_G(self, gamma, grad, eta, i):
        #return f.relu(gamma - eta * grad - eta * self.lambdas[i])
        return hard_ITML(gamma - eta * grad, self.ks[i])


def hard_ITML(x, k, non_neg=True):

    size_x = x.size()
    sign_x = x.sign()
    x = x.view(size_x[0], -1)

    if non_neg:
        max_values, idx_max_values = x.topk(k, dim=-1)
    else:
        max_values, idx_max_values = x.abs().topk(k, dim=-1)

    x = 0 * x
    x = x.scatter(-1, idx_max_values, max_values)
    x = x.view(size_x)
    x *= sign_x

    return x
'''
def get_eigenvalue(gamma_size, layers, max_it_eigen=50):

    if torch.cuda.is_available():
        rdn_data = torch.randn((1,) + gamma_size[1:]).cuda()
    else:
        rdn_data = torch.randn((1,) + gamma_size[1:])

    for idx in range(max_it_eigen):

        dx = layers.backward(rdn_data)
        res = layers.forward(dx)
        lamb = float(torch.norm(res, p=2))
        rdn_data = torch.div(res, lamb)

    return lamb
