import os
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import csr_matrix, vstack, coo_matrix
from tqdm import tqdm

from evaluation import *

def arr2coo(cols, values, shape):
    rows = torch.arange(cols.shape[0]).repeat_interleave(cols.shape[1])
    cols = cols.flatten()
    indices = torch.stack((rows, cols), dim=0)
    values = values.flatten()
    return torch.sparse.FloatTensor(indices, values, torch.Size(shape))

def fill_zeros(a):
    a = a.masked_fill_(a == 0, float('inf'))
    min_a = a.min(dim=1, keepdim=True)[0]
    a = a.where(a != float('inf'), min_a)
    return a

class Model:
    def __init__(self, network, num_experts, **kwargs):
        self.num_experts = num_experts
        self.devices, self.output_device = self.assign_devices()
        self.experts = nn.ModuleList( network(**kwargs).cuda(self.devices[i]) for i in range(self.num_experts) )
        
    def assign_devices(self):
        visible_devices = list(range(torch.cuda.device_count()))
        num_visible_devices = len(visible_devices)
        devices = [visible_devices[i % num_visible_devices] for i in range(self.num_experts)]
        output_device = visible_devices[-1]
        return devices, output_device
        
    def replicate(self, item):
        return [item.cuda(device) for device in self.devices]
        
    def gather(self, items):
        return [item.cuda(self.output_device) for item in items]
    
    def train(self, train_loader, test_loader, label_freq, inv_w, mlb, logger, model_dir,
              feature_size, label_size, lr, num_epochs, warm_up, clf_flood, div_flood, div_factor,
              print_step=100, top=5, **kwargs):
        
        optimizer = optim.Adam(self.experts.parameters(), lr)

        for epoch_idx in range(1, num_epochs+1):

            current_div_factor = div_factor if epoch_idx > warm_up else 0

            if current_div_factor > 0:
                weight = torch.Tensor( label_freq.clip(1) ** -0.5 ).cuda(self.output_device)
                _bce_loss = nn.BCELoss(weight=weight)
                clf_loss_fn = lambda p, y: _bce_loss(p, y) * y.shape[1] / weight.sum()
                temperature = 1 / weight
            else:
                clf_loss_fn = nn.BCELoss()
                temperature = 1
            
            div_loss_fn = nn.KLDivLoss(reduction='mean')

            for batch_idx, (ft_col, ft_value, sc_col, sc_value, lbl_col, lbl_value) in enumerate(train_loader, 1):
                feature = arr2coo(ft_col, ft_value, (len(ft_col), feature_size))
                score = arr2coo(sc_col, sc_value, (len(sc_col), label_size))
                label = arr2coo(lbl_col, lbl_value, (len(lbl_col), label_size))
            
                self.experts.train()
                logits, mean_logit = self.forward(feature, score)
                scores = self.replicate(score)

                probs = [torch.sigmoid(logits[i]) for i in range(self.num_experts)]
                dists = [F.log_softmax(logits[i] / temperature, dim=1) for i in range(self.num_experts)]
                with torch.no_grad():
                    mean_dist = F.softmax(mean_logit / temperature, dim=1)
                
                label = label.cuda(self.output_device).to_dense()
                
                total_clf_loss, total_div_loss = 0, 0
                for i in range(self.num_experts):
                    clf_loss =  clf_loss_fn(probs[i], label)
                    div_loss = -div_loss_fn(dists[i], mean_dist)
                    clf_loss = (clf_loss - clf_flood).abs() + clf_flood
                    div_loss = (div_loss - div_flood).abs() + div_flood
                    total_clf_loss += clf_loss
                    total_div_loss += div_loss

                optimizer.zero_grad()
                loss = total_clf_loss + current_div_factor * total_div_loss
                loss.backward()
                optimizer.step()
                
                if batch_idx % print_step == 0 or batch_idx == len(train_loader):
                    logger.info('epoch {0} {1} | clf loss: {cls_loss:.2e}  div loss: {div_loss:.2e}'.format(
                                epoch_idx, batch_idx * train_loader.batch_size,
                                cls_loss=total_clf_loss.item() / self.num_experts,
                                div_loss=total_div_loss.item() / self.num_experts))
            
            self.test(test_loader, inv_w, mlb, logger, None, feature_size, label_size, **kwargs)
            self.save_model(model_dir)

    def forward(self, feature, score):
        features, scores = self.replicate(feature), self.replicate(score)
        
        logits = []
        
        lock = threading.Lock()
        results = {}
        def _worker(i, device, expert, feature, score):
            try:
                with torch.cuda.device(device):
                    feature = feature.to_dense()
                    # score = score.to_dense()
                    score = fill_zeros(score.to_dense())
                    logit = expert(feature, score)
                with lock:
                    results[i] = logit
            except Exception as e:
                with lock:
                    results[i] = e
        
        threads = [threading.Thread(target=_worker,
                                    args=(i, device, expert, feature, score))
                  for i, (device, expert, feature, score) in
                  enumerate(zip(self.devices, self.experts, features, scores))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        for i in range(self.num_experts):
            result = results[i]
            if isinstance(result, Exception):
                raise result
            logits.append(result)
            
        logits = self.gather(logits)
        mean_logit = sum(logits) / len(logits)
        return logits, mean_logit
        
    def predict(self, data_loader, feature_size, label_size, top):
        preds = []
        
        for ft_col, ft_value, sc_col, sc_value in data_loader:
            feature = arr2coo(ft_col, ft_value, (len(ft_col), feature_size))
            score = arr2coo(sc_col, sc_value, (len(sc_col), label_size))
            
            self.experts.eval()
            with torch.no_grad():
                _, mean_logit = self.forward(feature, score)
                prob = torch.sigmoid(mean_logit)
                _, pred = torch.topk(prob, k=top)
            preds.append(pred.cpu())
        return np.concatenate(preds)
    
    def test(self, test_loader, inv_w, mlb, logger, model_dir, feature_size, label_size, **kwargs):
        self.load_model(model_dir)
        test_preds = self.predict(tqdm(test_loader, desc='Testing', leave=False), feature_size, label_size, top=5)
        test_labels = test_loader.dataset.labels
        
        p, n, psp, psn = [], [], [], []
        for k in (1, 3, 5):
            p.append(get_precision(test_preds, test_labels, mlb, top=k))
            n.append(get_ndcg(test_preds, test_labels, mlb, top=k))
            psp.append(get_psp(test_preds, test_labels, inv_w, mlb, top=k))
            psn.append(get_psndcg(test_preds, test_labels, inv_w, mlb, top=k))
        
        logger.info('P@1,3,5: %.2f, %.2f, %.2f' % tuple(p))
        logger.info('nDCG@1,3,5: %.2f, %.2f, %.2f' % tuple(n))
        logger.info('PSP@1,3,5: %.2f, %.2f, %.2f' % tuple(psp))
        logger.info('PSnDCG@1,3,5: %.2f, %.2f, %.2f' % tuple(psn))
    
    def save_model(self, model_dir):
        if model_dir != None:
            os.makedirs(model_dir, exist_ok=True)
            for i in range(self.num_experts):
                torch.save(self.experts[i].state_dict(), os.path.join(model_dir, f'{i}.pth'))
                
    def load_model(self, model_dir):
        if model_dir != None:
            for i in range(self.num_experts):
                self.experts[i].load_state_dict(torch.load(os.path.join(model_dir, f'{i}.pth')))
    

def test_baseline(test_loader, inv_w, mlb, logger):
    preds = []
    for _, _, sc_col, sc_value in tqdm(test_loader, desc='Testing baseline', leave=False):
        for col, value in zip(sc_col, sc_value):
            top_ind = torch.argsort(value, descending=True)[:5]
            preds.append(col[top_ind])
    test_preds = np.vstack(preds)
    test_labels = test_loader.dataset.labels
    
    p, n, psp, psn = [], [], [], []
    for k in (1, 3, 5):
        p.append(get_precision(test_preds, test_labels, mlb, top=k))
        n.append(get_ndcg(test_preds, test_labels, mlb, top=k))
        psp.append(get_psp(test_preds, test_labels, inv_w, mlb, top=k))
        psn.append(get_psndcg(test_preds, test_labels, inv_w, mlb, top=k))
    
    logger.info('P@1,3,5: %.2f, %.2f, %.2f' % tuple(p))
    logger.info('nDCG@1,3,5: %.2f, %.2f, %.2f' % tuple(n))
    logger.info('PSP@1,3,5: %.2f, %.2f, %.2f' % tuple(psp))
    logger.info('PSnDCG@1,3,5: %.2f, %.2f, %.2f' % tuple(psn))