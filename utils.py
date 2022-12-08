import logging
from shutil import ignore_patterns
import torch
import torch.nn.functional as F

def get_logger(filename,verbosity=1,name=None):
  level_dict={0:logging.DEBUG,1:logging.INFO,2:logging.WARNING}
  formatter=logging.Formatter(
      "%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s:\t%(message)s"
  )
  logger=logging.getLogger(name)
  logger.setLevel(level_dict[verbosity])

  fh=logging.FileHandler(filename,'w')
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  sh=logging.StreamHandler()
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  return logger

##This cell contain function to resize tensor for Cross Entropy loss
def normalize_size(tensor):
  ##Hàm chuẩn hóa size tensor lấy lại theo code B-MRC
    if len(tensor.size()) == 3:
        tensor = tensor.contiguous().view(-1, tensor.size(2))
    elif len(tensor.size()) == 2:
        tensor = tensor.contiguous().view(-1)

    return tensor

def calculate_A_O_loss(targets,logits,ifgpu=True,ignore_indexes=[],model_type=None):
  ##Hàm này tính loss cho aspect hay opinion
  ##Theo thống kê nhãn 0 nhiều gấp 8 lần nhãn 1 và gấp 16 lần nhãn 2 nên ta sẽ đánh weight theo thứ tự
  ##[1,2,4]
    gold_targets=normalize_size(targets)
    pred=normalize_size(logits)

    if ifgpu==True:
        weight = torch.tensor([1, 2, 4]).float().cuda()
    else:
        weight = torch.tensor([1, 2, 4]).float()

    loss=F.cross_entropy(pred,gold_targets.long(),ignore_index=-1,weight=weight)
    '''if 'deberta' not in model_type:
        gold_targets=normalize_size(targets)
        pred=normalize_size(logits)
        loss=F.cross_entropy(pred,gold_targets.long(),ignore_index=-1,weight=weight)
    else:
        loss=0
        for idx in range(len(ignore_indexes)):
            index=torch.tensor(ignore_indexes[idx])
            valid_index=(index == 0).nonzero(as_tuple=True)[0]
            target=targets[idx]
            pred=logits[idx]
            begin_target_idx = (target != -1).nonzero(as_tuple=True)[0][0]
            target=target[begin_target_idx.item():begin_target_idx.item()+len(index)]
            pred=pred[begin_target_idx.item():begin_target_idx.item()+len(index),:]
            valid_pred=pred[valid_index,:].unsqueeze(0)
            valid_target=target[valid_index]
            valid_pred=normalize_size(valid_pred)
            valid_target=normalize_size(valid_target)
            loss+=F.cross_entropy(valid_pred,valid_target.long(),ignore_index=-1,weight=weight)'''
    return loss

def sentiment_loss(pred_list,y_true_list,ignore_index=-1):
    epsilon=0.0000001
    lossS=0
    for i in range(len(pred_list)):
        loss=0
        for j in range(len(pred_list[i])):
            if pred_list[i][j]==ignore_index or y_true_list[i][j]==-ignore_index:
                continue
            loss+=((y_true_list[i][j]+epsilon)*torch.log(torch.tensor(pred_list[i][j]+epsilon)).item())
        lossS+=1/len(pred_list[i])*loss
    lossS=-1/(len(pred_list))*lossS
    return torch.tensor(lossS)

def is_one_exist(labels,ignore_index):
    '''
        Hàm giúp kiểm tra nếu có nhãn 1 trong labels hay không giúp quyết định bước xây dựng query tiếp theo
    '''
    if 1 not in labels:
        return False
    else:
        count=0
        one_index=(labels==1).nonzero(as_tuple=True)[0]
        for idx in one_index:
            if idx.item() in ignore_index:
                count+=1
        if count==len(one_index):
            return False
    return True
