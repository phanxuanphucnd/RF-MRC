# -*- coding: utf-8 -*-

import random
import torch
import torch.nn.functional as F
from utils import is_one_exist

##CLS_id=101 ##For BERT
#SEP_id=102 ##For BERT

CLS_id=1 ##For Deberta
SEP_id=2 ##For Deberta

def generating_batch_index(dataset,batch_size,shuffle=True,drop_last=False):
  '''
    Hàm này giúp xáo trộn và gom các index của các sample của một dataset thành các batch có args.batch_size
  '''
  idx_list=[i for i in range(dataset.__len__())]
  if shuffle==True:
    random.shuffle(idx_list)
  ##Slitting idx into batch
  start=0
  end=start+batch_size
  batch_idx=[]
  while end<len(idx_list):
    idx_sublist=idx_list[start:end]
    batch_idx.append(idx_sublist)
    start+=batch_size
    end+=batch_size
  if drop_last==False and len(idx_list)%batch_size!=0:
    idx_sublist=idx_list[start:len(idx_list)]
    batch_idx.append(idx_sublist)
  return batch_idx

def generating_one_query(sample,query_type='initial',max_len=None,outside_query=None,index=None,model_mode='train'):
  '''
    Hàm này giúp tạo một query từ một sample của ProccessedIDDataset:
      - query_type là loại câu hỏi. Có 3 dạng: initial, aspect, opinion. Riêng initial cần truyền thêm max_len
        để padding luôn
        + Hai dạng câu hỏi aspect và opinion nếu không được truyền outside_query sẽ lấy chính aspect hay opinion
         trong input làm query (dạng teacher forcing)
      - outside_query: là query nhập từ bên ngoài, với aspect hay opinion nó chính là top p aspect hay top q opinion
         sau mỗi bước question answering
      - index là chỉ số của sample trong dataset
      - model_mode: chế độ của model đang chạy. Nếu mà ở train hàm sẽ trả về ground truth answer cho việc
       tính loss ở bước tiếp theo. Còn ở chế độ khác train máy sẽ trả về answer full -1 và cũng không dùng
       answer này.
  '''
  if query_type=='initial':

    ##Generating input ids
    question=[CLS_id]+[SEP_id]+sample['text_ids']+[SEP_id]
    padding_num=max_len-len(question)
    ##padding
    input_ids=question+[0]*padding_num

    ##Generating attention mask
    attention_mask=[1]*len(question)+[0]*padding_num

    ##Generating token type ids
    token_type_ids=[0]*2+[1]*(len(question)-2+padding_num)

    ##Genearating aspect and opinion answer
    if model_mode=='train':
      aspect_answer=[-1]*2+sample['aspect_answer']+[0]+[-1]*padding_num
      opinion_answer=[-1]*2+sample['opinion_answer']+[0]+[-1]*padding_num
    else:
      aspect_answer=[-1]*len(input_ids)
      opinion_answer=[-1]*len(input_ids)

    assert len(input_ids)==len(attention_mask)==len(token_type_ids)==len(aspect_answer)==len(opinion_answer)
    return input_ids,attention_mask,token_type_ids,aspect_answer,opinion_answer
  elif query_type=='aspect':
    if outside_query==None and index!=None:
      query=sample['aspect_questions_ids'][index]
    else:
      query=outside_query
    context=sample['texts_ids'][index]
    temp_answer=sample['opinion_answers'][index]
    input_ids=[CLS_id]+query+[SEP_id]+context+[SEP_id]
    attention_mask=[1]*len(input_ids)
    token_type_ids=[0]*(len(query)+2)+[1]*(len(context)+1)
    if model_mode=='train':
      answer=[-1]*(len(query)+2)+temp_answer+[0]
    else:
      answer=[-1]*len(input_ids)
    assert len(input_ids)==len(attention_mask)==len(token_type_ids)==len(answer)
    return input_ids,attention_mask,token_type_ids,answer
  elif query_type=='opinion':
    if outside_query==None and index!=None:
      query=sample['opinion_questions_ids'][index]
    else:
      query=outside_query
    context=sample['texts_ids'][index]
    temp_answer=sample['aspect_answers'][index]
    input_ids=[CLS_id]+query+[SEP_id]+context+[SEP_id]
    attention_mask=[1]*len(input_ids)
    token_type_ids=[0]*(len(query)+2)+[1]*(len(context)+1)
    if model_mode=='train':
      answer=[-1]*(len(query)+2)+temp_answer+[0]
    else:
      answer=[-1]*len(input_ids)
    assert len(input_ids)==len(attention_mask)==len(token_type_ids)==len(answer)
    return input_ids,attention_mask,token_type_ids,answer

def genrating_batch_data(batch_unprocess_data,istrain=True,max_len=None,ifgpu=True):
  '''
    Hàm này chủ yếu dùng để bắt các initial query, initial_query_mask và initial_query_seg vào thành
    các tensor với shape: [batch_size,tokens+padding]
    Nếu không sinh cho dữ liệu train (istrain=False) hàm sẽ không trả về answers
  '''
  initial_queries=[]
  initial_query_masks=[]
  initial_query_segs=[]
  initial_aspect_answers=[]
  initial_opinion_answers=[]

  for sample in batch_unprocess_data:
    ###Generating input ids, attention_mask and token type ids
    input_ids,attention_mask,token_type_ids,aspect_answer,opinion_answer=generating_one_query(sample,query_type='initial',max_len=max_len)
    initial_queries.append(input_ids)
    initial_query_masks.append(attention_mask)
    initial_query_segs.append(token_type_ids)
    if istrain==True:
      initial_aspect_answers.append(aspect_answer)
      initial_opinion_answers.append(opinion_answer)
  if istrain==True:
    if ifgpu==True:
      return torch.tensor(initial_queries).cuda(),torch.tensor(initial_query_masks).cuda(),torch.tensor(initial_query_segs).cuda(),torch.tensor(initial_aspect_answers).cuda(),torch.tensor(initial_opinion_answers).cuda()
    else:
      return torch.tensor(initial_queries),torch.tensor(initial_query_masks),torch.tensor(initial_query_segs),torch.tensor(initial_aspect_answers),torch.tensor(initial_opinion_answers)
  else:
    if ifgpu==True:
      return torch.tensor(initial_queries).cuda(),torch.tensor(initial_query_masks).cuda(),torch.tensor(initial_query_segs).cuda()
    else:
      return torch.tensor(initial_queries),torch.tensor(initial_query_masks),torch.tensor(initial_query_segs)

def generating_model_dataset(dataset,batch_size,shuffle=True,drop_last=False,istrain=True,ifgpu=True):
  '''
    Hàm tạo dữ liệu ban đầu cho train, dev và test data cho vào model
      + Đồng thời hàm cũng sẽ đếm max len cho một initial query.
    Nếu không tạo dữ liệu cho train hàm cũng trả về bỏ bớt initial_aspect_answer và initial_opinion_answers
      (hai câu trả lời đã được padding theo size của query)
      + Ở dạng khác train (dev hoặc test) hàm vẫn trả về aspect, opinion và sentiments label chưa padding để
       tiện cho việc đánh giá model
  '''
  batch_idxes=generating_batch_index(dataset,batch_size,shuffle=shuffle,drop_last=drop_last)
  for batch_idx in batch_idxes:
    sample_list=[]
    texts_ids=[]
    texts=[]
    aspect_questions_ids=[]
    opinion_questions_ids=[]
    aspect_answers=[]
    opinion_answers=[]
    sentiments=[]
    ignore_indexes = [] ##Những vị trí bỏ qua nếu sử dụng mô hình pretrained dạng BPE

    max_len=0
    for idx in batch_idx:
      sample=dataset.getIndexSample(idx)
      sample_list.append(sample)
      texts.append(sample['text'])
      texts_ids.append(sample['text_ids'])
      aspect_questions_ids.append(sample['aspect_question_ids'])
      opinion_questions_ids.append(sample['opinion_question_ids'])
      aspect_answers.append(sample['aspect_answer'])
      opinion_answers.append(sample['opinion_answer'])
      sentiments.append(sample['sentiment'])
      ignore_indexes.append(sample['ignore_index'])

      if len(sample['text_ids'])+3>max_len:
        max_len=len(sample['text_ids'])+3
    if istrain==True:
      initial_queries,initial_masks,initial_segs,initial_aspect_answers,initial_opinion_answers=genrating_batch_data(sample_list,istrain=istrain,max_len=max_len,ifgpu=ifgpu)
      yield {
          'texts': texts,
          'texts_ids':texts_ids,
          'initial_input_ids':initial_queries,
          'initial_attention_mask':initial_masks,
          'initial_token_type_ids':initial_segs,
          'initial_aspect_answers':initial_aspect_answers,
          'initial_opinion_answers':initial_opinion_answers,
          'aspect_questions_ids':aspect_questions_ids,
          'opinion_questions_ids':opinion_questions_ids,
          'aspect_answers':aspect_answers,
          'opinion_answers':opinion_answers,
          'sentiments':sentiments,
          'ignore_indexes':ignore_indexes
      }
    elif istrain==False:
      initial_queries,initial_masks,initial_segs=genrating_batch_data(sample_list,istrain=istrain,max_len=max_len,ifgpu=ifgpu)
      yield {
          'texts': texts,
          'texts_ids':texts_ids,
          'initial_input_ids':initial_queries,
          'initial_attention_mask':initial_masks,
          'initial_token_type_ids':initial_segs,
          'aspect_answers':aspect_answers,
          'opinion_answers':opinion_answers,
          'sentiments':sentiments,
          'ignore_indexes':ignore_indexes
      }

def padding_query_batch(input_ids_list,attention_mask_list,token_type_ids_list,answer_list,max_len=None,ifgpu=True):
  '''
    Hàm này dùng padding một batch gồm các query theo size của query lớn nhất trong batch
    Trong notebook này code này chỉ dùng cho padding aspect và opinion query cho dữ liệu train
  '''
  for i in range(len(input_ids_list)):
    ##Padding length
    padding_len=max_len-len(input_ids_list[i])

    ##input_ids
    input_ids_list[i].extend([0]*padding_len)

    ##attention mask
    attention_mask_list[i].extend([0]*padding_len)

    ##token type ids
    token_type_ids_list[i].extend([1]*padding_len)

    ##answer
    answer_list[i].extend([-1]*padding_len)

  if ifgpu==True:
    return torch.tensor(input_ids_list).cuda(),torch.tensor(attention_mask_list).cuda(),torch.tensor(token_type_ids_list).cuda(),torch.tensor(answer_list).cuda()
  else:
    return torch.tensor(input_ids_list),torch.tensor(attention_mask_list),torch.tensor(token_type_ids_list),torch.tensor(answer_list)

def generating_next_query(batch_dict,logits,last_queries,args,query_type='aspect',model_mode='train'):
  '''
    Hàm này sinh dữ liệu query cho bước tiếp theo của multi hop, query_type là biến quyết định sẽ sinh
      ra dạng câu hỏi nào cho query type
    + Hướng xử lý trường hợp sau khi softmax không sinh ra nhãn 1-(begin aspect hay begin opinion)
      -   Nếu đang trong quá trình train - sử dụng chính ground truth làm query tiếp theo (teacher forcing)
      - Trường hợp model ở mode khác train, sẽ lấy top p vị trí có xác suất gán nhãn 1 cao nhất (tuy nhiên 
        softmax lại không dán nhãn 1 vì xác suất của 0 hoặc 2 cao hơn).
          + Nếu dãy nhãn không full 2 thì vẫn lấy kèm nhãn hai nếu nó ngay sau nhãn 1 xem như nhãn I
          + Nếu dãy nhãn là full 2 thì chỉ lấy những thằng nhãn 1. 
  '''
  prob=F.softmax(logits,dim=-1)
  top_val,top_ind=torch.max(prob,dim=-1)
  input_ids_list=[]
  attention_mask_list=[]
  token_type_ids_list=[]
  answer_list=[]
  max_len=0
  for idx,ind_tensor in enumerate(top_ind):
    passenge_index = (last_queries[idx]==SEP_id).nonzero(as_tuple=True)[0]
    passenge_index = torch.tensor([num for num in range(passenge_index[0].item()+1,passenge_index[1].item())],dtype=torch.long).unsqueeze(1)
    labels=ind_tensor[passenge_index].squeeze(1)
    if 'deberta' in args.model_type:
      index=torch.tensor(batch_dict['ignore_indexes'][idx])
      ignore_index=(index == -1).nonzero(as_tuple=True)[0]
    else:
      ignore_index=torch.tensor([])
    ##Xử lý khi 1 không có trong labels của bước multi hop hiện tại
    if is_one_exist(labels,ignore_index)==False:
      if model_mode=='train':
        input_ids,attention_mask,token_type_ids,answer=generating_one_query(batch_dict,query_type=query_type,index=idx)
      else:
        outside_query=[]
        prob_i=prob[idx].transpose(0,1)[1]
        passenge_prob_i=prob_i[passenge_index].squeeze(1) ##Trích chọn xác suất ở những ký tự của câu có nhãn 1
        _,one_index=torch.sort(passenge_prob_i,descending=True)
        new_top_val=top_val[idx][passenge_index].squeeze(1)
        ##Kiểm tra nếu không có cả nhãn 0 và 1 nghĩa là full nhãn 2
        two_index=torch.tensor([])
        count=0
        for index in one_index:
          inde=index.item()
          if inde in ignore_index:
            continue
          outside_query.append(batch_dict['texts_ids'][idx][inde])
          count+=1
          inde+=1
          while inde < len(batch_dict['texts_ids'][idx]) and inde in ignore_index:
            outside_query.append(batch_dict['texts_ids'][idx][inde])
            inde+=1
          if query_type=='aspect':
            if count>=args.p:
              break
          else:
            if count>=args.q:
              break
        input_ids,attention_mask,token_type_ids,answer=generating_one_query(batch_dict,query_type=query_type,index=idx,outside_query=outside_query,model_mode=model_mode)
      if len(input_ids)>max_len:
        max_len=len(input_ids)
      input_ids_list.append(input_ids)
      attention_mask_list.append(attention_mask)
      token_type_ids_list.append(token_type_ids)
      answer_list.append(answer)
    ##Nếu có nhãn 1 thì lấy top p hoặc q tương ứng với aspect hay query như bình thường
    else:
      outside_query=[]
      new_top_val=top_val[idx][passenge_index].squeeze(1)
      one_index=(labels == 1).nonzero(as_tuple=True)[0]
      two_index=(labels == 2).nonzero(as_tuple=True)[0]
      _top_val,_top_ind=torch.sort(new_top_val,descending=True)
      count=0
      for index in _top_ind:
        inde=index.item()
        if inde in two_index or inde in ignore_index:
          continue
        if inde in one_index:
          outside_query.append(batch_dict['texts_ids'][idx][inde])
          count+=1
          inde+=1
          while inde < len(batch_dict['texts_ids'][idx]) and (inde in two_index or inde in ignore_index):
            outside_query.append(batch_dict['texts_ids'][idx][inde])
            inde+=1
          if query_type=='aspect':
            if count>=args.p:
              break
          else:
            if count>=args.q:
              break
        else:
          continue
      ##outside_query=outside_query[:-1]
      input_ids,attention_mask,token_type_ids,answer=generating_one_query(batch_dict,query_type=query_type,index=idx,outside_query=outside_query,model_mode=model_mode)
      if len(input_ids)>max_len:
        max_len=len(input_ids)
      input_ids_list.append(input_ids)
      attention_mask_list.append(attention_mask)
      token_type_ids_list.append(token_type_ids)
      answer_list.append(answer)
  inputs_ids,attention_masks,token_type_ids,answers=padding_query_batch(input_ids_list,attention_mask_list,token_type_ids_list,answer_list,max_len=max_len,ifgpu=args.ifgpu)
  return inputs_ids,attention_masks,token_type_ids,answers