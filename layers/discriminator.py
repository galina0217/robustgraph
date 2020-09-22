import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_h, critic='bilinear', dataset=None, attack_model=True):
        super(Discriminator, self).__init__()
        self.critic = critic

        if self.critic == 'bilinear' or (dataset == 'citeseer' and attack_model==False):
        # if self.critic == 'bilinear':
            self.f_k = nn.Bilinear(n_h, n_h, 1)
            for m in self.modules():
                self.weights_init(m)

        # self.weight1 = nn.Parameter(
        #         torch.FloatTensor(n_h, 100))
        # self.weight2 = nn.Parameter(
        #         torch.FloatTensor(n_h, 100))

        # torch.nn.init.xavier_uniform(self.weight1)
        # torch.nn.init.xavier_uniform(self.weight2)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)    # (1, 1, 512)
        c_x = c_x.expand_as(h_pl)    # (1, 2708, 512)

        # seperable
        # c_x = c_x.squeeze().mm(self.weight2)
        # sc_1 = h_pl.squeeze().mm(self.weight1)
        # sc_2 = h_mi.squeeze().mm(self.weight1)
        #
        # sc_1 = sc_1.mm(c_x.t())[:,0].unsqueeze(0)   # (1, 2708)
        # sc_2 = sc_2.mm(c_x.t())[:,0].unsqueeze(0)    # (1, 2708)

        if self.critic == 'bilinear':
            ## bilinear
            sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)    # (1, 2708)
            sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)    # (1, 2708)

            if s_bias1 is not None:
                sc_1 += s_bias1
            if s_bias2 is not None:
                sc_2 += s_bias2

        ## multilinear
        # sc_1 = h_pl.squeeze().mm(self.weight1).mm(self.weight2).mm(c_x.squeeze().t())[:,0].unsqueeze(0)
        # sc_2 = h_mi.squeeze().mm(self.weight1).mm(self.weight2).mm(c_x.squeeze().t())[:,0].unsqueeze(0)

        elif self.critic == 'inner product':
        ## inner product
            sc_1 = h_pl.bmm(c_x.transpose(1,2))[:,:,0]   # (1, 2708)
            sc_2 = h_mi.bmm(c_x.transpose(1,2))[:,:,0]   # (1, 2708)

        logits = torch.cat((sc_1, sc_2), 1)    # (1, 5416)

        return logits


class Discriminator_jsd(nn.Module):
    def __init__(self, n_h):
        super(Discriminator_jsd, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)    # (1, 1, 512)
        c_x = c_x.expand_as(h_pl)    # (1, 2708, 512)
        #
        # sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)    # (1, 2708)
        # sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)    # (1, 2708)
        #
        # if s_bias1 is not None:
        #     sc_1 += s_bias1
        # if s_bias2 is not None:
        #     sc_2 += s_bias2
        #
        # logits = torch.cat((sc_1, sc_2), 1)    # (1, 5416)


        # Outer product, we want a N x N x n_local x n_multi tensor.
        u = h_pl.bmm(c_x.transpose(1,2))

        # Since we have a big tensor with both positive and negative samples, we need to mask.
        mask = torch.eye(u.shape[-1])
        if torch.cuda.is_available():
            mask = mask.cuda()
        n_mask = 1 - mask

        # Positive term is just the average of the diagonal.
        # E_pos = (u.mean(2) * mask).sum() / mask.sum()
        u = u.squeeze()
        # u = u.mean(2).mean(2)
        # E_pos = (u.mean(2) * mask).sum() / mask.sum()
        E_pos = (u * mask).sum((0, 1)) / mask.sum()

        E_neg = torch.logsumexp(u * n_mask, (0, 1)) - torch.log(n_mask.sum((0, 1)))
        loss = - E_neg + E_pos

        return loss

class Discriminator2(nn.Module):
    def __init__(self, ft_size, n_h):
        super(Discriminator2, self).__init__()
        self.f_k = nn.Bilinear(ft_size, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)  # (1, 2708)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)  # (1, 2708)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)  # (1, 5416)

        return logits

