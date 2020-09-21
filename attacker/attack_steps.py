"""
**For most use cases, this can just be considered an internal class and
ignored.**

This module contains the abstract class AttackerStep as well as a few subclasses. 

AttackerStep is a generic way to implement optimizers specifically for use with
:class:`robustness.attacker.AttackerModel`. In general, except for when you want
to :ref:`create a custom optimization method <adding-custom-steps>`, you probably do not need to
import or edit this module and can just think of it as internal.
"""

import torch as ch
import numpy as np

def bisection(a, eps, xi, ub=1):
    pa = np.clip(a, 0, ub)
    # print('{}/{}'.format(np.sum(pa)/2, eps))
    if np.sum(pa) <= eps:
        # print('np.sum(pa) <= eps !!!!')
        upper_S_update = pa
    else:
        mu_l = np.min(a - 1)
        mu_u = np.max(a)
        # mu_a = (mu_u + mu_l)/2
        while np.abs(mu_u - mu_l) > xi:
            # print('|mu_u - mu_l|:',np.abs(mu_u - mu_l))
            mu_a = (mu_u + mu_l) / 2
            gu = np.sum(np.clip(a - mu_a, 0, ub)) - eps
            gu_l = np.sum(np.clip(a - mu_l, 0, ub)) - eps
            # print('gu:',gu)
            if gu == 0:
                print('gu == 0 !!!!!')
                break
            if np.sign(gu) == np.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a

        upper_S_update = np.clip(a - mu_a, 0, ub)

    return upper_S_update


class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, A, nb_nodes, eps, step_size, eps_x=0.1, step_size_x=1e-5, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.A = A
        self.nb_nodes = nb_nodes
        self.eps = eps
        self.eps_x = eps_x
        self.step_size = step_size
        self.step_size_x = step_size_x
        self.use_grad = use_grad

        self.xi = 1e-5

        # self.C = 1 - 2 * self.A
        # self.upper_S_0 = ch.autograd.Variable(ch.zeros((self.nb_nodes, self.nb_nodes), dtype=ch.float32), requires_grad=True)

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set

        Args:
            ch.tensor x : the input to project back into the feasible set.

        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).

        Parameters:
            g (ch.tensor): the raw gradient

        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, delta_A, show=False):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        delta_A = delta_A.detach().cpu().numpy()
        while (1):
            randm = np.random.uniform(size=(self.nb_nodes, self.nb_nodes))
            ret = np.where(delta_A > randm, 1, 0)
            if show:
                b = np.triu(ret, 1).sum()
                print('b/eps: {}/{}'.format(b, self.eps))
            ret = np.triu(ret, 1) + np.triu(ret, 1).T
            if ch.cuda.is_available():
                return ch.FloatTensor(ret).cuda()
            return ch.FloatTensor(ret)


    # def to_image(self, x, x_ori):
    #     '''
    #     Given an input (which may be in an alternative parameterization),
    #     convert it to a valid image (this is implemented as the identity
    #     function by default as most of the time we use the pixel
    #     parameterization, but for alternative parameterizations this functino
    #     must be overriden).
    #     '''
    #     x = x.detach().cpu().numpy()
    #     x_ori = x_ori.detach().cpu().numpy()
    #     while(1):
    #         randm = np.random.uniform(size=(self.nb_nodes, self.nb_nodes))
    #         ret = np.where(x > randm, 1, 0)
    #         b = sum(sum(ret - x_ori))/2
    #         # b = len(np.where(np.triu(ret, 1))[0])
    #         print('b/eps: {}/{}'.format(2*b, self.eps))
    #         ret = np.triu(ret, 1) + np.triu(ret, 1).T
    #         return ch.FloatTensor(ret).cuda()
    #
    #         # randm = np.random.uniform(size=(self.nb_nodes, self.nb_nodes))
    #         # ret = np.where(x > randm, 1, 0)
    #         # b = len(np.where(np.triu(ret, 1))[0])
    #         # print('b/eps: {}/{}'.format(b, self.eps))
    #         # if b == self.eps:
    #         #     ret = np.triu(ret, 1) + np.triu(ret, 1).T
    #         #     return ch.FloatTensor(ret).cuda()
    #         #
    #         # idx = np.dstack(np.unravel_index(np.argsort(np.triu(x, 1).ravel()),
    #         #                                  (self.nb_nodes, self.nb_nodes)))[:,-int(self.eps/2):]
    #         # ret = np.zeros((self.nb_nodes, self.nb_nodes))
    #         # ret[idx[:, :, 0], idx[:, :, 1]] = [1 for i in range(idx.shape[1])]
    #         # ret = np.triu(ret, 1) + np.triu(ret, 1).T


### Instantiations of the AttackerStep class

# L-0 threat model
class L0Step(AttackerStep):
    """
    Attack step for :math:`\ell_0` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_0 \leq \epsilon\}
    """
    def project(self, a):
        ub = 1
        pa = ch.clamp(a, 0, ub)
        if pa.sum() <= self.eps*2:
            a_proj = pa
        else:
            mu_l = ch.min(a - 1)
            mu_u = ch.max(a)
            while abs(mu_u - mu_l) > self.xi:
                mu_a = (mu_u + mu_l) / 2
                gu = ch.clamp(a - mu_a, 0, ub).sum() - self.eps*2
                gu_l = ch.clamp(a - mu_l, 0, ub).sum() - self.eps*2
                if gu == 0:
                    print('gu == 0 !!!!!')
                    break
                if ch.sign(gu) == ch.sign(gu_l):
                    mu_l = mu_a
                else:
                    mu_u = mu_a
            a_proj = ch.clamp(a - mu_a, 0, ub)

        a_proj_ret = ch.triu(a_proj, 1) + ch.triu(a_proj, 1).T
        return a_proj_ret

    def step(self, delta_x, g):
        """
        """
        delta_x = delta_x + g * self.step_size
        # step = ch.sign(g) * self.step_size
        return delta_x

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (ch.rand_like(x) - 0.5) * self.eps
        return ch.clamp(new_x, 0, 1)


# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x, orig_x):
        """
        """
        diff = x - orig_x
        diff = ch.clamp(diff, -self.eps_x, self.eps_x)
        return ch.clamp(diff + orig_x, 0, 1)

    def step(self, x, g):
        """
        """
        step = ch.sign(g) * self.step_size_x
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (ch.rand_like(x) - 0.5) * self.eps_x
        return ch.clamp(new_x, 0, 1)
