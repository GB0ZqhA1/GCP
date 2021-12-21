import torch
import math, random
from tqdm import tqdm
from model import *

def prune_reg(model, decay):
    reg = 0.0
    for p in model.parameters():
        if hasattr(p, 'groups'):
            o,i,k,_ = p.size()
            ord_group = torch.cat(p.groups)
            ord_p = p[ord_group].view(len(p.groups), -1, i, k, k)
            reg += torch.sum(torch.sqrt(torch.sum(ord_p**2,(1,3,4))))*((o/len(p.groups))**0.5)
    return decay*reg


class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.input_size = 1
        for i in input[0].size():
            self.input_size*=i
    
    def close(self):
        self.hook.remove()

class GCP:
    def __init__(self, network, ratio, threshold, numgroup, mingroup=1):
        self.ratio = 1 - ratio - 0.001
        self.threshold = threshold
        self.numgroup = numgroup
        self.groups = []
        self.inds = []
        self.blocks = []
        self.hooks = []
        self.mingroup = mingroup
        
        for m in network.modules():
            if isinstance(m, ResNetBasicblock):
                self.blocks.append([m.conv_b.weight,m.conv_a.weight])
                self.hooks.append([Hook(m.conv_b), Hook(m.conv_a)])
            elif isinstance(m, wide_basic):
                self.blocks.append([m.conv2.weight,m.conv1.weight])
                self.hooks.append([Hook(m.conv2), Hook(m.conv1)])
        
        network(torch.randn(1,3,32,32).to(next(network.parameters()).device))
        
        input_sizes=[]
        for h in self.hooks:
            for i in h:
                input_sizes.append(i.input_size)
                i.close()
        min_input = min(input_sizes)
        print('Group Settings')
        for b, h in zip(self.blocks, self.hooks):
            b[-1].numgroup = max(1, numgroup * min_input // h[-1].input_size)
            for l, i in zip(b[:-1], h[:-1]):
                l.numgroup = max(2, numgroup * min_input // i.input_size)
            print(*[l.numgroup for l in b])

    def alt_opt(self, chnorm, neginds, numgroup, comp_rate):
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
                groups = [[] for _ in range(numgroup)]
                for no, och in enumerate(sample):
                    groups[och].append(no)
                new_norm = torch.stack([chnorm[groups[n],].sum(dim=0) for n in range(numgroup)])
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
            grth = grnorm.sort(dim=3)[0][:, :, :, int(i * comp_rate)].unsqueeze_(3)  # n_init x 1 x n_group x 1
            neginds = (grnorm < grth).float()  # n_init x 1 x n_group x i
        group_final+=group_list
        grnorm_final+=grnorm_list
        grnorm = torch.stack(grnorm_final)  # n_init x n_group x i
        infloss = grnorm.topk(int(i * comp_rate), dim=2, largest=False)[0].sum(dim=2).sum(dim=1)

        group_list = list(zip(infloss, group_final))

        return min(group_list, key=lambda x:x[0])

    def hashvalue(self, inds):
        a = ""
        for t in inds:
            a += str(t)
        return a

    def cluster(self, p, comp_rate, seeds):
        chunk = 256
        p.ind = torch.zeros_like(p.data.sum(3, keepdim=True).sum(2, keepdim=True))
        nextind = p.nextind
        
        o, i, c, _ = p.data.size()
        chnorm = p.data.view(o, i, -1).pow(2).sum(dim=2)
        chnorm.mul_(nextind.view(-1,1)+1e-8*(1-nextind.view(-1,1)))
        
        ngroup = p.numgroup
        chth = chnorm.sort(dim=1)[0][:, int(i * comp_rate)].unsqueeze_(1)
        chinds = (chnorm < chth).float()  # o x i

        # mask initialization without duplication
        unique_inds = []
        ind_hash = []
        neginds = []
        for ci in chinds:
            unique = True
            if ci.sum()<len(ci):
                for u in unique_inds:
                    if torch.equal(ci, u):
                        unique = False
                        break
            else:
                unique=False
            if unique:
                unique_inds.append(ci)
        tries = 0
        while len(ind_hash)<seeds:
            if tries>seeds*2:
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
        infloss, groups = self.alt_opt(chnorm,neginds[0:min(num_inds,chunk)], ngroup, comp_rate)
        for ni in range(num_inds//chunk-1):
            newinfloss, newgroups = self.alt_opt(chnorm, neginds[(ni+1)*chunk:min(num_inds,(ni+2)*chunk)], ngroup, comp_rate)

            if newinfloss<infloss:
                infloss=newinfloss
                groups=newgroups

        groups = [torch.cuda.LongTensor(g) for g in groups]
        grnorm = torch.stack([chnorm.index_select(0,g).sum(0) for g in groups])
        grth = grnorm.sort(dim=1)[0][:, int(i * comp_rate)].view(ngroup, 1)

        negind = (grnorm < grth).view(1, ngroup, i).float()
        infloss = chnorm.view(o, 1, i).mul(negind).sum(dim=2)

        # group optimization
        transloss, transition = infloss.sort(dim=1)
        loss, sample = transloss.tolist(), transition.tolist()
        on = [0 for _ in range(o)]
        changed = True
        groups = [[] for _ in range(ngroup)]
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
        grnorm = torch.stack([chnorm.index_select(0,g).sum(0) for g in groups])
        infloss = grnorm.topk(int(i * comp_rate), dim=1, largest=False)[0].sum()
        grth = grnorm.sort(dim=1)[0][:, int(i * comp_rate)].view(ngroup, 1)
        ind = (grnorm >= grth).view(ngroup, i, 1, 1).float()
        for n in range(ngroup):
            p.ind.index_copy_(0, p.groups[n], ind[n].view(1, i, 1, 1).expand(len(p.groups[n]), i, 1, 1))
        prune_mask = p.ind*p.nextind.view(-1,1,1,1)
        p.comp_rate = prune_mask.mean()
        weight_loss = ((p.data**2)*(1-prune_mask)).sum().item()
        return weight_loss, (weight_loss/(p.data**2).sum()).item()

    def initialize(self, dynamic=True):
        totalloss = 0
        th = self.threshold
        bar = tqdm(total=len(self.blocks))
        layer_wise_cr = []
        layer_count=0
        for b in self.blocks:
            nextind = torch.ones_like(b[0].data.view(b[0].size(0),-1).sum(dim=1))
            for p in b:
                layer_count+=1
                p.nextind = nextind
                if dynamic:
                    lower, upper, cur = 0, 1, 0.5
                    for _ in range(10):
                        weight_loss, rel_loss = self.cluster(p, cur, 256)
                        if rel_loss>th:
                            upper = cur
                            cur = (lower+cur)/2
                        elif rel_loss<th:
                            lower = cur
                            cur = (upper+cur)/2
                else:
                    cur = self.ratio
                weight_loss, rel_loss = self.cluster(p, cur, 4096)
                print('CR: %.1f'%((1-p.comp_rate.item())*100,))
                layer_wise_cr.append(1-p.comp_rate.item())
                totalloss += weight_loss
                nextind = (p.ind.sum(0)>0).float().view(-1)
                self.groups.append(p.groups)
                self.inds.append(p.ind)
                bar.set_description("[pruning_loss:%.4f]" % (totalloss**0.5,))
            bar.update()
        bar.close()
        
        return totalloss, layer_wise_cr

    @torch.no_grad()
    def prune(self):
        for b in self.blocks:
            for p in b:
                p.mul_(p.ind)
