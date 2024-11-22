import os
import numpy as np
import time
import torch
import argparse
import logging
from utils import StandardScaler, DataLoaderST_TaGNN, masked_mae_loss, masked_mae_loss_
from TaGNN import TaGNNModel
from metrics import RMSE_MAE_R2

def print_model(model):
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    logger.info(f'In total: {param_count} trainable parameters.')
    return

def get_model():
    model = TaGNNModel(gpu=args.gpu, 
                     temperature=args.temperature, 
                     cl_decay_steps=args.cl_decay_steps, 
                     filter_type=args.filter_type, 
                     max_diffusion_step=args.max_diffusion_step,
                     num_nodes=args.num_nodes,
                     num_rnn_layers=args.num_rnn_layers,
                     rnn_units=args.rnn_units,
                     input_dim=args.input_dim,
                     output_dim=args.output_dim,
                     horizon=args.horizon,
                     seq_len=args.seq_len,
                     use_curriculum_learning=args.use_curriculum_learning, 
                     dim_fc=args.dim_fc).to(device)
    return model

def prepare_x_y(x, y):
    x = x.permute(0, 3, 2, 1) 
    y = y[..., :args.output_dim]
    return x.to(device), y.to(device)

def compute_loss(y_true, y_predicted):
    return masked_mae_loss(y_predicted, y_true) 

def compute_loss_(y_true, y_predicted, score = None):
    return masked_mae_loss_(y_predicted, y_true, score)

def evaluate(fold, model, Data, mode, batches_seen, train_feas, adj_mx, scaler):
    with torch.no_grad():
        model = model.eval()
        losses, ys_true, ys_pred = [], [], []
        if mode =='valid':
            datasetsx = Data.valid[0]
            datasetsy = Data.valid[1]
            datasetsa = Data.valid[2]
            datasetst = Data.valid[3]
        else:
            datasetsx = Data.test[0]
            datasetsy = Data.test[1]
            datasetsa = Data.test[2]
            datasetst = Data.test[3]

        for X, Y, A, T in Data.get_batches(datasetsx, datasetsy, datasetsa, datasetst, args.batch_size, True):
            testx = X.to(device)
            testx = testx.transpose(1, 2)
            testy = Y.to(device)
            testadj = A.to(device)
            testtend = T.to(device)
            x, y  = prepare_x_y(testx, testy)
            output, (loss_n, loss_a), mid_output, pone_score = model(x, train_feas, testtend, testadj)
            pind = (pone_score[0] - pone_score[1]).softmax(dim = -1)
            y_pred = output[:,:,-1,:]
            y_pred = y_pred.view(y.shape[0], model.nhead, y_pred.shape[-2], y_pred.shape[-1])
            y_pred = torch.einsum('bhfl,bh->bfl', (y_pred, pind))
            y_pred = y_pred.squeeze(dim = 2)
            y_pred = scaler.inverse_transform_(y_pred)
            y_true = y
            loss_1 = compute_loss(y_pred, y_true)
            y_true = y_true.cpu().numpy() 
            y_pred = y_pred.cpu().numpy()
            losses.append((loss_1.item()))
            ys_true.append(y_true)
            ys_pred.append(y_pred)
        val_loss = np.mean(losses)
        ys_true = ys_true[:-1]
        ys_pred = ys_pred[:-1]
        ys_true = np.vstack(ys_true) 
        ys_pred = np.vstack(ys_pred)
        if mode == 'test':
            rmse, mae, r2 = RMSE_MAE_R2(ys_true, ys_pred)
            logger.info('Horizon 1: loss: {:.4f}, mae: {:.4f}, rmse: {:.4f}, r2: {:.4f}'.format(val_loss, mae, rmse, r2))
            return val_loss, ys_true, ys_pred, r2, mae, rmse
        else:
            return val_loss, ys_true, ys_pred
        
def traintest_model(fold, Data, train_feas, adj_mx, scaler, adj_factor = 0.1, noise_factor = 0.1, align_factor = 0.1):
    model = get_model()
    modelpt_path = f'{path}/{model_name}_{timestring}_{fold}.pt'
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, eps=args.epsilon)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    min_val_loss = float('inf')
    wait = 0
    batches_seen = 0
    for epoch_num in range(args.epochs):
        start_time = time.time()
        model = model.train()
        losses = []
        for X, Y, A, T in Data.get_batches(Data.train[0], Data.train[1], Data.train[2], Data.train[3], args.batch_size, True):
            optimizer.zero_grad()
            trainx = X.to(device)
            trainx = trainx.transpose(1, 2)
            trainy = Y.to(device)
            testadj = A.to(device)
            testtend = T.to(device)
            x, y = prepare_x_y(trainx, trainy)
            output, (loss_n, loss_a), mid_output, (po_score, ne_score) = model(x, train_feas, testtend, testadj)
            loss_n = loss_n.view(y.shape[0], model.nhead)
            y_pred = output[:,:,-1,:].squeeze(dim = 2)
            y_pred = scaler.inverse_transform_(y_pred)
            loss_1 = compute_loss_(y_pred.view(y.shape[0], model.nhead, y.shape[1]), y.unsqueeze(dim = 1).expand(y.shape[0], model.nhead, y.shape[1]), (po_score - ne_score).clone().detach_())
            pred = mid_output.view(mid_output.shape[0], mid_output.shape[-1] * mid_output.shape[-2])
            adj_mx_ = testadj[:,:-1,:-1].unsqueeze(dim = 1).expand(testadj.shape[0], model.nhead, mid_output.shape[-2], mid_output.shape[-1])
            true_label = adj_mx_.flatten(0,1).view(-1, mid_output.shape[-1] * mid_output.shape[-1])
            bce_loss = torch.nn.BCELoss()
            loss_g = bce_loss(pred, true_label)
            loss = loss_1 + adj_factor * loss_g + noise_factor * loss_n.mean() + align_factor * loss_a
            losses.append((loss_1.item()+loss_g.item()+loss_n.mean().item()))
            batches_seen += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        train_loss = np.mean(losses)
        lr_scheduler.step()
        val_loss, _, _ = evaluate(fold, model, Data, 'valid', batches_seen, train_feas, adj_mx, scaler)
        end_time2 = time.time()
        message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch_num + 1, 
                   args.epochs, batches_seen, train_loss, val_loss, optimizer.param_groups[0]['lr'], (end_time2 - start_time))
        logger.info(message)
        
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
        elif val_loss >= min_val_loss:
            wait += 1
            if wait == args.patience:
                logger.info('Early stopping at epoch: %d' % epoch_num)
                break
    
    logger.info('=' * 35 + 'Best model performance' + '=' * 35)
    model.load_state_dict(torch.load(modelpt_path))
    model.eval()
    test_loss, _, _, r2, mae, rmse = evaluate(fold, model, Data, 'test', batches_seen, train_feas, adj_mx, scaler)

    return test_loss, r2, mae, rmse

#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['XIHU'], default='XIHU', help='which dataset to run')
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the ratio of training and validation data among the total')
parser.add_argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--num_nodes', type=int, default=39, help='num_nodes')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--horizon', type=int, default=1, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=6, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=5, help='number of output channel')
parser.add_argument('--num_rnn_layers', type=int, default=1, help='number of rnn layers')
parser.add_argument('--rnn_units', type=int, default=64, help='number of rnn units')
parser.add_argument('--filter_type', type=str, default='dual_random_walk', help='filter_type')
parser.add_argument('--max_diffusion_step', type=int, default=3, help='max_diffusion_step')
parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
parser.add_argument("--epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--patience", type=int, default=10, help="patience used for early stop")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--base_lr", type=float, default=0.005, help="base learning rate")
parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
parser.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning")
parser.add_argument("--steps", type=eval, default=[20, 30, 40], help="steps")
parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
parser.add_argument('--knn_k', type=int, default=10, help='knn_k')
parser.add_argument('--dim_fc', type=int, default=672, help='dim_fc')
parser.add_argument('--test_every_n_epochs', type=int, default=5, help='test_every_n_epochs')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
args = parser.parse_args()

model_name = 'TaGNN'
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'/data2/tongshuo/GTS_MegaCRN-main/newgcmt_save/{args.dataset}_{model_name}_{timestring}'
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
if not os.path.exists(path): os.makedirs(path)
    
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
class MyFormatter(logging.Formatter):
    def format(self, record):
        spliter = ' '
        record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
        record.args = tuple()
        return super().format(record)
formatter = MyFormatter()
handler = logging.FileHandler(logging_path, mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

logger.info('model', model_name)
logger.info('dataset', args.dataset)
logger.info('trainval_ratio', args.trainval_ratio)
logger.info('val_ratio', args.val_ratio)
logger.info('num_nodes', args.num_nodes)
logger.info('seq_len', args.seq_len)
logger.info('horizon', args.horizon)
logger.info('input_dim', args.input_dim)
logger.info('output_dim', args.output_dim)
logger.info('num_rnn_layers', args.num_rnn_layers)
logger.info('rnn_units', args.rnn_units)
logger.info('loss', args.loss)
logger.info('epochs', args.epochs)
logger.info('batch_size', args.batch_size)
logger.info('base_lr', args.base_lr)
logger.info('use_curriculum_learning', args.use_curriculum_learning)
logger.info('knn_k', args.knn_k)
logger.info('dim_fc', args.dim_fc)

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
#####################################################################################################
data_paths = [
    ".../data/target1.xlsx",
    ".../data/target2.xlsx",
    ".../data/target3.xlsx",
    ".../data/target4.xlsx",
    ".../data/target5.xlsx",
    ]
def main():
    loss, rmse, mae, r2 = [],[],[],[],[]
    for fold in range(5):
        print("------------------------------------FOLD", {fold+1}, "------------------------------------")
        Data = DataLoaderST_TaGNN(num=fold, file_name='.../data/XIHU_Data_Noise.npz',trend_name='.../data/XIHU_Trend_Data_Noise.npz', data_paths=data_paths, device=device, horizon = args.horizon, window= args.seq_len)
        scaler = Data.scalar
        Data.train[0][:, :-1, :, :5] = scaler.transform(Data.train[0][:, :-1, :, :5])
        Data.valid[0][:, :-1, :, :5] = scaler.transform(Data.valid[0][:, :-1, :, :5])
        Data.test[0][:, :-1, :, :5] = scaler.transform(Data.test[0][:, :-1, :, :5])
        df = Data.train[0][:,:,:12,:5]
        train_feas = df[:int(df.shape[0]*args.trainval_ratio*(1 - args.val_ratio))]
        scaler1 = StandardScaler(mean=train_feas.mean(), std=train_feas.std())
        train_feas = scaler1.transform(train_feas)
        train_feas = train_feas.transpose(0, 1).flatten(-2, -1).mean(dim = 1)
        adj_mx = None

        train_feas = torch.Tensor(train_feas).to(device)
        logger.info('train xs.shape, ys.shape', Data.train[0].shape, Data.train[1].shape)
        logger.info('val xs.shape, ys.shape', Data.valid[0].shape, Data.valid[1].shape)
        logger.info('test xs.shape, ys.shape', Data.test[0].shape, Data.test[1].shape)

        test_loss, r2_all, mae_all, rmse_all = traintest_model(fold+1, Data, train_feas, adj_mx, scaler)
        loss.append(test_loss)
        rmse.append(rmse_all)
        mae.append(mae_all)
        r2.append(r2_all)
    logger.info(args.dataset, 'training and testing ended', time.ctime())
    
    print('\n\nResults for 5 Folds\n\n')
    print('valid\tR2\tMAE\tRMSE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(r2),np.mean(mae),np.mean(rmse)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(r2),np.std(mae),np.std(rmse)))

if __name__ == '__main__':
    main()

