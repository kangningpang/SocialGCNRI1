import world
import torch
import torch.nn as nn
import torch.nn.functional as F


class PureBPR(nn.Module):
    def __init__(self, config, dataset):
        super(PureBPR, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        print("using Normal distribution initializer")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss


class fft(nn.Module):
    def __init__(self, args):
        super(fft, self).__init__()
        # self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(0.5)
        # self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self,input_tensor):
        input_tensor = input_tensor.T
        seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        a, b = x.shape
        complex_weight = nn.Parameter(torch.randn(1, b, a, 2, dtype=torch.float32) * 0.02)
        weight = torch.view_as_complex(complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')

        out_dropout = nn.Dropout(0.5)
        hidden_states = out_dropout(sequence_emb_fft)
        hidden_states = hidden_states + input_tensor
        hidden_states = hidden_states.T
        return hidden_states



def pearson_correlation(x, y):
        # 计算x和y的均值
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        y_mean = torch.mean(y, dim=-1, keepdim=True)

        # 计算x和y的偏差
        x_dev = x - x_mean
        y_dev = y - y_mean

        # 计算皮尔逊相关系数
        numerator = torch.sum(x_dev * y_dev, dim=-1)
        denominator = torch.sqrt(torch.sum(x_dev ** 2, dim=-1)) * torch.sqrt(torch.sum(y_dev ** 2, dim=-1))

        # 防止除以零
        denominator = torch.where(denominator < 1e-7, torch.tensor(1e-7).to(denominator.device), denominator)

        r = numerator / denominator

        return r

class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self._init_weight()

    def _init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['layer']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.f = nn.Sigmoid()
        self.interactionGraph,self.interactionGraph2= self.dataset.getInteractionGraph()
        print(f"{world.model_name} is already to go")


    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        users_emb=fft(users_emb)
        items_emb=fft(items_emb)
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        embs2 = [all_emb]
        G = self.interactionGraph
        G2= self.interactionGraph2
        # G3= self.interactionGraph3

        for layer in range(self.n_layers+1):
            if layer==0:
               all_emb = torch.sparse.mm(G, all_emb)
               # embs.append(all_emb)
               embs2.append(all_emb)
            else:
               # a = F.cosine_similarity(embs[-1], embs[0], dim=-1)
               # a=pearson_correlation(embs[-1], embs[0],dim=-1)
               # b = F.cosine_similarity(embs2[-1], embs[0], dim=-1)
               b=pearson_correlation(embs2[-1], embs2[0],dim=-1)
               # all_embeddings = torch.einsum('a,ab->ab', a, torch.sparse.mm(G, embs[-1]))
               # all_emb = torch.einsum('a,ab->ab', a, torch.sparse.mm(G, embs[-1])) + torch.einsum('a,ab->ab', (1-a), embs[0])
               all_emb2 =  torch.einsum('a,ab->ab', b, torch.sparse.mm(G2, embs2[-1])) + torch.einsum('a,ab->ab', (1-b), embs2[0])
               # embs.append(all_emb)
               embs2.append(all_emb2)

        # embs = torch.stack(embs, dim=1)
        embs2 = torch.stack(embs2, dim=1)
        # embs=(embs+embs2)/2
        # print(embs.size())
        light_out = torch.mean(embs2, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.final_user, self.final_item
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss


class SocialLGN(LightGCN):
    def _init_weight(self):
        super(SocialLGN, self)._init_weight()
        self.socialGraph = self.dataset.getSocialGraph()
        self.Graph_Comb = Graph_Comb(self.latent_dim)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        # users_emb=fft(users_emb)
        # items_emb=fft(items_emb)
        all_emb = torch.cat([users_emb, items_emb])
        # user_0, item_0 = torch.split(all_emb, [self.num_users, self.num_items])
        A = self.interactionGraph
        A2=self.interactionGraph2
        S = self.socialGraph
        embs = [all_emb]
        for layer in range(self.n_layers):
            # embedding from last layer
            users_emb, items_emb = torch.split(all_emb, [self.num_users, self.num_items])
            # social network propagation(user embedding)
            users_emb_social = torch.sparse.mm(S, users_emb)
            # user-item bi-network propagation(user and item embedding)
            all_emb_interaction = torch.sparse.mm(A, all_emb)
            # get users_emb_interaction
            users_emb_interaction, items_emb_next = torch.split(all_emb_interaction, [self.num_users, self.num_items])
            # graph fusion model
            users_emb_next = self.Graph_Comb(users_emb_social, users_emb_interaction)
            all_emb = torch.cat([users_emb_next, items_emb_next])
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        final_embs = torch.mean(embs, dim=1)
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items


class Graph_Comb(nn.Module):  #图融合操作，结合U-I图嵌入和社交嵌入
    def __init__(self, embed_dim):
        super(Graph_Comb, self).__init__()
        self.att_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.comb = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, y):
        h1 = torch.tanh(self.att_x(x))
        h2 = torch.tanh(self.att_y(y))
        output = self.comb(torch.cat((h1, h2), dim=1))
        output = output / output.norm(2)
        return output
