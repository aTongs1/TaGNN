import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from layer import dilated_inception, mixprop, LayerNorm
import random

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mae_loss_(y_pred, y_true, score = None):
    score = score.softmax(dim = -1).unsqueeze(dim = -1)
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    if score is not None:
        loss = torch.einsum('bhc,bhl->bcl',(loss,score))
    return loss.mean()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def sample_gumbel(device, shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(device, logits, temperature, eps=1e-10):
    sample = sample_gumbel(device, logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(device, logits, temperature, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(device, logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.gpu = int(model_kwargs.get('gpu', 0))
        self.device = torch.device("cuda:{}".format(self.gpu)) if torch.cuda.is_available() else torch.device("cpu")


class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, a_unc = True, \
                 static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, \
                    conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, \
                        in_dim=6, out_dim=5, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true 
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.skip_channels = skip_channels
        self.a_unc = a_unc
        self.device = device
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        if not self.gcn_true:
            self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                else:
                    self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.a_unc:
            self.end_conv_2_ss = nn.Conv2d(in_channels=end_channels,
                                                out_channels=out_dim,
                                                kernel_size=(1,1),
                                                bias=True)
            
            self.end_conv_1_sigma = nn.Conv2d(in_channels=skip_channels,
                                                out_channels=end_channels,
                                                kernel_size=(1,1),
                                                bias=True)
            self.end_conv_2_sigma = nn.Conv2d(in_channels=end_channels,
                                                out_channels=out_dim,
                                                kernel_size=(1,1),
                                                bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)
        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, adp, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'
        ss_step = input[:,:-1,:-1,-2:-1]
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))
        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(-1,-2))
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x_feas = F.relu(skip)
        x = F.relu(self.end_conv_1(x_feas))
        if self.a_unc:
            x_log_sigma_sq = F.relu(self.end_conv_1_sigma(x_feas))
            x_log_sigma_sq = self.end_conv_2_sigma(x_log_sigma_sq)[:,:,:-1,:]
            x_ss = self.end_conv_2_ss(x)[:,:,:-1,:]
            x = self.end_conv_2(x)
            loss_ss = (torch.exp(-x_log_sigma_sq)*(x_ss - ss_step)**2)/2.0 + x_log_sigma_sq/2.0
            loss_ss = loss_ss.squeeze().mean(dim = -1).mean(dim = -1)
        else:
            x = self.end_conv_2(x)
            loss_ss = torch.zeros(x.shape[0]).to(self.device)
        return x, x_feas.squeeze(dim = -1), loss_ss

class NodeAlignment(nn.Module):
    def __init__(self, emb_dim1, emb_dim2, device, nnode = 38, nhead = 4, hidden_dim = 64):
        super().__init__()

        self.emb_dim1 = emb_dim1
        self.emb_dim2 = emb_dim2
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.nnode = nnode
        self.device = device
        self.lin1 = nn.Linear(self.emb_dim1, self.hidden_dim)
        self.lin2 = nn.Linear(self.emb_dim2, self.hidden_dim)
        self.indexes = [i for i in range(self.nnode)]
        self.loss_fn = nn.CrossEntropyLoss()

    def gen_neg(self, data2): 
        neg_data = []
        bs = data2.shape[0]
        for bid in range(bs):
            sind = [i for i in range(self.nnode)]
            random.shuffle(sind)
            while (torch.tensor(self.indexes) == torch.tensor(sind)).any():
                random.shuffle(sind)
            neg_data.append(data2[bid][sind,:].unsqueeze(dim = 0))
        neg_data = torch.cat(neg_data, dim = 0)
        return neg_data
    
    def cal_score(self, data1, data2):
        return torch.einsum('bnlc,bncw->bnlw',(data1.unsqueeze(dim = -2), data2.unsqueeze(dim = -1)))

    def forward(self, data1, data2):
        data1 = F.relu(self.lin1(data1))
        data2 = F.relu(self.lin2(data2))
        ne_data = self.gen_neg(data2)
        po_score = self.cal_score(data1, data2) 
        ne_score = self.cal_score(data1, ne_data) 
        po_prob = po_score.flatten(0,1).sigmoid().squeeze(dim = -1)
        ne_prob = ne_score.flatten(0,1).sigmoid().squeeze(dim = -1)
        prob = torch.cat([po_prob, ne_prob], dim = 0)
        prob = torch.cat([1-prob, prob], dim = -1)
        label = torch.cat([torch.ones(po_prob.shape), torch.zeros(ne_prob.shape)],dim = 0).long().to(self.device)
        loss = self.loss_fn(prob, label.squeeze())
        return po_score.squeeze().mean(dim = -1), ne_score.squeeze().mean(dim = -1), loss


class TaGNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)

        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self.temperature = float(model_kwargs.get('temporature', 0.5))
        self.dim_fc = int(model_kwargs.get('dim_fc', False))
        self.embedding_dim = 100
        self.nhead = 4
        self.a_unc = True
        self.pe = -1
        self.time_inception = gtnet(gcn_true = True, buildA_true = True, gcn_depth = 2, num_nodes = 39, a_unc=self.a_unc, device = self.device, predefined_A=None, \
                 static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, \
                    conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, \
                        in_dim=6, out_dim=5, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)
        
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1) 
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1) 
        self.hidden_drop = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(int((self.embedding_dim * 4)/self.nhead), self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
            return labels_onehot
        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.num_nodes-1, self.num_nodes-1])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        self.rel_send = torch.FloatTensor(rel_send).to(self.device)

        self.tconv1 =  nn.Conv1d(5, 16, 6, stride = 1)
        self.tconv2 = nn.Conv1d(16, 32, 5, stride = 1)
        self.tconv3 = nn.Conv1d(32, self.embedding_dim, 3, stride = 1)
        self.tbn1 = torch.nn.BatchNorm1d(16)
        self.tbn2 = torch.nn.BatchNorm1d(32)
        self.tbn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.pelin = nn.Linear(self.num_nodes-1, self.embedding_dim) if self.pe<0 else nn.Linear((self.num_nodes-1)*(self.pe+1), self.embedding_dim)
        
        self.sal1 = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead = self.nhead, dim_feedforward = self.embedding_dim, dropout = 0, batch_first = True)
        self.sal2 = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead = self.nhead, dim_feedforward = self.embedding_dim, dropout = 0, batch_first = True)
        self.fuse_lin = nn.Linear(2*self.embedding_dim, 2*self.embedding_dim)
        self.alignment = NodeAlignment(emb_dim1 = self.time_inception.skip_channels, emb_dim2 = self.embedding_dim, device = self.device, nnode=self.num_nodes-1, nhead = self.nhead)
    
    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, inputs, node_feas, tend_feas, adjs, batches_seen=None):
        bs = inputs.shape[0] 
        x = node_feas.view(self.num_nodes, 1, -1) 
        x = self.conv1(x) 
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(self.num_nodes, -1) 
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = x.unsqueeze(dim = 0).expand(bs, self.num_nodes, -1)
        
        xt = tend_feas.transpose(-1,-2)  
        xt = F.relu(self.tconv1(xt.flatten(0,1)))
        xt = self.tbn1(xt)
        xt = F.relu(self.tconv2(xt))
        xt = self.tbn2(xt)
        xt = F.relu(self.tconv3(xt))
        xt = self.tbn3(xt)
        xt = xt.view(bs, self.num_nodes-1, -1)

        adjs_ = adjs[:,:-1,:-1]
        if self.pe < 0:
            I = torch.eye(adjs_.shape[1], adjs_.shape[2]).to(self.device)
            evecs = []
            for i in range(bs):
                adjs_sum = adjs_[i].sum(dim=1)
                adjs_sum[adjs_sum == 0] = 1e-10 
                D_sqrt = torch.diag(1/torch.sqrt(adjs_sum))
                adj_ = I - torch.matmul(torch.matmul(D_sqrt, adjs_[i].float()), D_sqrt)
                (_,evecs_) = torch.linalg.eig(adj_)
                evecs.append(evecs_.real.unsqueeze(dim = 0))
            evecs = torch.cat(evecs, dim = 0)
        else:
            evecs = [torch.eye(self.num_nodes-1).unsqueeze(dim = 0).expand(bs, self.num_nodes-1, self.num_nodes-1).to(self.device)]
            rwps = []
            for i in range(bs):
                d_1 = torch.diag(1.0/(adjs_[i].sum(dim = 1)))
                rwp = torch.matmul(d_1, adjs_[i].float())
                rwps.append(rwp.unsqueeze(dim = 0))
            rwps = torch.cat(rwps, dim = 0)
            for hop in range(self.pe):
                evecs.append(torch.bmm(evecs[-1], rwps))
            evecs = torch.cat(evecs, dim = -1)

        xt = self.sal1(xt + self.pelin(evecs))
        xt = self.sal2(xt)
        xt_feas = xt.clone().view(bs, self.num_nodes-1, -1)
        x = self.fuse_lin(torch.cat([x[:,:-1,:], xt], dim = -1))
        x = x.view(x.shape[0], x.shape[1], self.nhead, -1).transpose(1,2)
        x = x.flatten(0, 1)
        
        trel_rec = self.rel_rec.unsqueeze(dim = 0).expand((x.shape[0],)+self.rel_rec.shape)
        trel_send = self.rel_send.unsqueeze(dim = 0).expand((x.shape[0],)+self.rel_send.shape)
        receivers = torch.bmm(trel_rec, x) 
        senders = torch.bmm(trel_send, x) 
        x = torch.cat([senders, receivers], dim=-1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        padj = gumbel_softmax(self.device, x, temperature=self.temperature, hard=True)
        padj = padj[:, :, 0].clone().reshape(padj.shape[0], self.num_nodes-1, -1)
        mask = torch.eye(self.num_nodes-1, self.num_nodes-1).bool().to(self.device)
        padj = padj.masked_fill_(mask, 0).view(bs, -1, self.num_nodes-1, self.num_nodes-1)
        exp_adjs = adjs.unsqueeze(dim = 1).expand(bs, padj.shape[1], self.num_nodes, self.num_nodes)
        padjs = torch.cat([torch.cat([padj, exp_adjs[:,:,-2:-1,:-1]], dim = -2), exp_adjs[:,:,:,-2:-1]], dim = -1)

        inputs = inputs.unsqueeze(dim = 1).expand(inputs.shape[0], self.nhead, inputs.shape[1], inputs.shape[2], inputs.shape[3])
        outputs, x_feas, loss_ss = self.time_inception(inputs.flatten(0,1), padjs.flatten(0, 1))
        x_feas = x_feas.view(bs, self.nhead, -1, self.num_nodes)[:,:,:,:-1].transpose(-1,-2)
        xt_feas = xt_feas.unsqueeze(dim = 1).expand(bs, self.nhead, self.num_nodes-1, -1)
        po_score, ne_score, loss_align = self.alignment(x_feas.flatten(0,1), xt_feas.flatten(0,1))
        po_score = po_score.view(bs, self.nhead, -1)
        ne_score = ne_score.view(bs, self.nhead, -1)

        return outputs.sigmoid(), (loss_ss, loss_align), x.softmax(-1)[:,:, 0].clone().reshape(x.shape[0], self.num_nodes-1, -1), (po_score.squeeze(), ne_score.squeeze())