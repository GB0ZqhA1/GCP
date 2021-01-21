import torch
from torch.optim.optimizer import Optimizer, required
import random
from sklearn.cluster import KMeans
from tqdm import tqdm

class SGD_GCP(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, ratio=0, prune=True, numgroup=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(SGD_GCP, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, "position"):
                    p.ind = torch.ones_like(p.data.sum(2).sum(2)).view(p.data.size(0), p.data.size(1), 1, 1)
                    p.groups = None

        self.prune = prune
        self.ratio = 1 - ratio
        self.numgroup = numgroup

    def __setstate__(self, state):
        super(SGD_GCP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def alt_opt(self, chnorm, neginds):
        o, i = chnorm.data.size()
        prevloss = [-1 for _ in range(neginds.size(0))]
        grnorm_final = []
        group_final = []
        for iteration in range(32):
            infloss = chnorm.view(1, o, 1, i).mul(neginds).sum(dim=3)  # n_init x o x n_group
            transloss, transition = infloss.min(dim=2)  # n_init x o x n_group, n_init x o x n_group
            infloss = transloss.sum(dim=1).tolist()

            group_list = []
            grnorm_list = []
            new_loss = []
            for iternum, sample in enumerate(transition.tolist()):
                groups = [[] for _ in range(self.numgroup)]
                for no, och in enumerate(sample):
                    groups[och].append(no)
                new_norm = torch.stack([chnorm[groups[n],].sum(dim=0) for n in range(self.numgroup)])
                if infloss[iternum] != prevloss[iternum]:
                    new_loss.append(infloss[iternum])
                    group_list.append(groups)
                    grnorm_list.append(new_norm)
                else:
                    group_final.append(groups)
                    grnorm_final.append(new_norm)
            prevloss = new_loss
            if len(grnorm_list)==0:
                break
            grnorm = torch.stack(grnorm_list).unsqueeze_(1)  # n_init x 1 x n_group x i
            grth = grnorm.sort(dim=3)[0][:, :, :, int(i * self.ratio) - 1].unsqueeze_(3)  # n_init x 1 x n_group x 1
            neginds = (grnorm <= grth).float()  # n_init x 1 x n_group x i
        group_final+=group_list
        grnorm_final+=grnorm_list
        grnorm = torch.stack(grnorm_final)  # n_init x n_group x i
        infloss = grnorm.topk(int(i * self.ratio), dim=2, largest=False)[0].sum(dim=2).sum(dim=1)

        group_list = list(zip(infloss, group_final))

        return min(group_list, key=lambda x:x[0])

    def hashvalue(self, inds):
        a = ""
        for t in inds:
            a += str(t)
        return a
    def kmeans(self):
        totalloss = 0
        for group in self.param_groups:
            bar = tqdm(total=len(group['params']))
            for p in group['params']:
                if hasattr(p, "position"):
                    o, i, c, _ = p.data.size()
                    chnorm = p.data.view(o, i, -1).pow(2).sum(dim=2)
                    ngroup = min(i, self.numgroup)

                    kmeans = KMeans(n_clusters=ngroup, init="random", n_init=8192).fit(p.data.view(o,-1).abs().cpu().numpy())
                    groups = [[] for _ in range(self.numgroup)]
                    for och, chgr in enumerate(kmeans.labels_):
                        groups[chgr].append(och)

                    groups = [torch.cuda.LongTensor(g) for g in groups]
                    grnorm = torch.stack([chnorm.index_select(0, g).mean(0) if len(g) > 0
                                          else chnorm.index_select(0, g).sum(0) for g in groups])
                    grth = grnorm.sort(dim=1)[0][:, int(i * self.ratio)].view(self.numgroup, 1)

                    negind = (grnorm <= grth).view(1, self.numgroup, i).float()
                    infloss = chnorm.view(o, 1, i).mul(negind).sum(dim=2)  # o x n_group
                    transloss, transition = infloss.sort(dim=1)  # o x n_group, o x n_group
                    loss, sample = transloss.tolist(), transition.tolist()
                    on = [0 for _ in range(o)]
                    changed = True
                    groups = [[] for _ in range(self.numgroup)]
                    for no, och in enumerate(sample):
                        groups[och[0]].append(no)
                    while changed:
                        changed = False
                        tr = []
                        for gr in groups:
                            if len(gr) > o // ngroup:
                                changed = True
                                nextloss = []
                                for och in gr:
                                    nextloss.append((och, loss[och][on[och] + 1] - loss[och][on[och]]))
                                nextloss = min(nextloss, key=lambda x: x[1])
                                o_tr, cost = nextloss
                                tr.append((o_tr, cost, sample[o_tr][on[o_tr]], sample[o_tr][on[o_tr] + 1]))
                        if changed:
                            ch, _, prevgr, nextgr = min(tr, key=lambda x: x[1])
                            on[ch] += 1
                            groups[prevgr].remove(ch)
                            groups[nextgr].append(ch)

                    groups = [torch.cuda.LongTensor(g) for g in groups]
                    p.groups = groups
                    grnorm = torch.stack([chnorm.index_select(0, g).mean(0) if len(g) > 0
                                          else chnorm.index_select(0, g).sum(0) for g in groups])
                    infloss = grnorm.topk(int(i * self.ratio), dim=1, largest=False)[0].sum()
                    totalloss += infloss.item()
                    grth = grnorm.sort(dim=1)[0][:, int(i * self.ratio)].view(self.numgroup, 1)
                    ind = (grnorm >= grth).view(self.numgroup, i, 1, 1).float()
                    for n in range(self.numgroup):
                        p.ind.index_copy_(0, p.groups[n], ind[n].view(1, i, 1, 1).expand(len(p.groups[n]), i, 1, 1))

                    p.data.mul_(p.ind)

                    bar.set_description("[infloss:%.4f]" % (totalloss,))
                bar.update()
            bar.close()

        return totalloss

    def cluster(self):
        totalloss = 0
        for group in self.param_groups:
            bar = tqdm(total=len(group['params']))
            for p in group['params']:
                if hasattr(p, "position"):
                    starts = 8192
                    chunk = 256
                    o, i, c, _ = p.data.size()
                    chnorm = p.data.view(o, i, -1).pow(2).sum(dim=2)
                    ngroup = min(i, self.numgroup)
                    chth = chnorm.sort(dim=1)[0][:, int(i * self.ratio) - 1].unsqueeze_(1)
                    chinds = (chnorm <= chth).float()  # o x i

                    # mask initialization without duplication
                    unique_inds = []
                    ind_hash = []
                    neginds = []
                    for ci in chinds:
                        unique = True
                        for u in unique_inds:
                            if torch.equal(ci, u):
                                unique = False
                                break
                        if unique:
                            unique_inds.append(ci)
                    tries = 0
                    while len(ind_hash)<starts:
                        if tries>starts*2:
                            break
                        initlist = torch.randperm(len(unique_inds))[:ngroup].tolist()
                        initlist.sort()
                        tries+=1
                        v = self.hashvalue(initlist)
                        if v not in ind_hash:
                            ind_hash.append(v)
                            indlist = [unique_inds[u] for u in initlist]
                            while len(indlist) < ngroup:
                                indlist.append(unique_inds[random.randint(0, len(unique_inds) - 1)])
                            neginds.append(torch.stack(indlist))
                        indlist = [unique_inds[u] for u in initlist]
                        while len(indlist) < ngroup:
                            indlist.append(unique_inds[random.randint(0, len(unique_inds) - 1)])
                        neginds.append(torch.stack(indlist))
                    neginds = torch.stack(neginds).unsqueeze_(1)
                    num_inds = len(ind_hash)

                    # alternating optimization
                    infloss, groups = self.alt_opt(chnorm,neginds[0:min(num_inds,chunk)])
                    for ni in range(num_inds//chunk-1):
                        newinfloss, newgroups = self.alt_opt(chnorm, neginds[(ni+1)*chunk:min(num_inds,(ni+2)*chunk)])

                        if newinfloss<infloss:
                            infloss=newinfloss
                            groups=newgroups

                    groups = [torch.cuda.LongTensor(g) for g in groups]
                    grnorm = torch.stack([chnorm.index_select(0,g).sum(0) for g in groups])
                    grth = grnorm.sort(dim=1)[0][:, int(i * self.ratio)].view(self.numgroup, 1)

                    negind = (grnorm <= grth).view(1, self.numgroup, i).float()
                    infloss = chnorm.view(o, 1, i).mul(negind).sum(dim=2)

                    # group optimization
                    transloss, transition = infloss.sort(dim=1)
                    loss, sample = transloss.tolist(), transition.tolist()
                    on = [0 for _ in range(o)]
                    changed = True
                    groups = [[] for _ in range(self.numgroup)]
                    for no, och in enumerate(sample):
                        groups[och[0]].append(no)
                    while changed:
                        changed = False
                        tr = []
                        for gr in groups:
                            if len(gr) > o // ngroup:
                                changed = True
                                nextloss = []
                                for och in gr:
                                    nextloss.append((och, loss[och][on[och] + 1] - loss[och][on[och]]))
                                nextloss = min(nextloss, key=lambda x: x[1])
                                o_tr, cost = nextloss
                                tr.append((o_tr, cost, sample[o_tr][on[o_tr]], sample[o_tr][on[o_tr] + 1]))
                        if changed:
                            ch, _, prevgr, nextgr = min(tr, key=lambda x: x[1])
                            on[ch] += 1
                            groups[prevgr].remove(ch)
                            groups[nextgr].append(ch)

                    # masking
                    groups = [torch.cuda.LongTensor(g) for g in groups]
                    p.groups = groups
                    grnorm = torch.stack([chnorm.index_select(0,g).sum(0) for g in groups])
                    infloss = grnorm.topk(int(i * self.ratio), dim=1, largest=False)[0].sum()
                    totalloss += infloss.item()
                    grth = grnorm.sort(dim=1)[0][:, int(i * self.ratio)].view(self.numgroup, 1)
                    ind = (grnorm >= grth).view(self.numgroup, i, 1, 1).float()
                    for n in range(self.numgroup):
                        p.ind.index_copy_(0, p.groups[n], ind[n].view(1, i, 1, 1).expand(len(p.groups[n]), i, 1, 1))

                    p.data.mul_(p.ind)

                    bar.set_description("[pruning_loss:%.4f]" % (totalloss**0.5,))
                bar.update()
            bar.close()

        return totalloss

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if (weight_decay != 0):
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

                if self.prune:
                    if hasattr(p, "position"):
                        p.mul_(p.ind)

        return loss


class SGD_GL(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(SGD_GL, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_GL, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def reg(self, rate):
        loss=0
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, "position"):
                    loss += p.norm(dim=3).norm(dim=2).sum()
        return loss*rate

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                if (weight_decay != 0):
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss
