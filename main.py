from statistics import mode
from utils import get_logger
import torch
from torch.autograd import Variable
from model import RFMRC
#from model_deberta import RFMRC
import os
from torch.optim import Adam
from transformers import get_scheduler,AdamW,get_linear_schedule_with_warmup
from dataset_support import generating_model_dataset
import argparse

def test(model,batch_generator,standard,logger=None):
  model.eval()

  aspect_match_num=0
  aspect_predict_num=0
  aspect_target_num=0

  opinion_match_num=0
  opinion_predict_num=0
  opinion_target_num=0

  sentiment_match_num=0
  sentiment_predict_num=0
  sentiment_target_num=0

  asp_pol_match_num=0
  asp_pol_predict_num=0
  asp_pol_target_num=0

  for batch_index,batch_dict in enumerate(batch_generator):
    aspect_terms,opinion_terms,sentiments,_,_,_=model(batch_dict,model_mode='test')
    ##config ASC predicts and targets
    sent_targets=[]
    sent_labels=[]
    asp_pol_pair_labels=[]
    for asp_pol in standard[batch_index]['asp_pol_target']:
      sent_targets.append([asp_pol[0],asp_pol[-1]])
      sent_labels.append([asp_pol[0],sentiments[0][asp_pol[0]]])
      
    '''for idx in range(len(sentiments[0])):
      if sentiments[0][idx]!=-1:
        sent_labels.append([idx,sentiments[0][idx]])'''

    for asp in aspect_terms[0]:
      asp_pol_pair_labels.append(asp+[sentiments[0][asp[0]]])
    
    ##Count aspect
    aspect_predict_num+=len(aspect_terms[0])
    aspect_target_num+=len(standard[batch_index]['asp_target'])
    for asp_pred in aspect_terms[0]:
      for asp_tar in standard[batch_index]['asp_target']:
        if asp_pred==asp_tar:
          aspect_match_num+=1

    ##Count opinion
    opinion_predict_num+=len(opinion_terms[0])
    opinion_target_num+=len(standard[batch_index]['opi_target'])
    for opi_pred in opinion_terms[0]:
      for opi_tar in standard[batch_index]['opi_target']:
        if opi_pred==opi_tar:
          opinion_match_num+=1

    ##Count sentiment
    sentiment_predict_num+=len(sent_labels)
    sentiment_target_num+=len(sent_targets)
    for sent_pred in sent_labels:
      for sent_tar in sent_targets:
        if sent_pred==sent_tar:
          sentiment_match_num+=1
    
    ##Overall
    asp_pol_predict_num+=len(asp_pol_pair_labels)
    asp_pol_target_num+=len(standard[batch_index]['asp_pol_target'])
    for asp_pol_pred in asp_pol_pair_labels:
      for asp_pol_tar in standard[batch_index]['asp_pol_target']:
        if asp_pol_pred==asp_pol_tar:
          asp_pol_match_num+=1

  ##Calculate F1-score for aspect:
  aspect_precision = float(aspect_match_num) / float(aspect_predict_num)
  aspect_recall = float(aspect_match_num) / float(aspect_target_num)
  if (aspect_precision + aspect_recall)!=0:
    aspect_f1 = 2 * aspect_precision * aspect_recall / (aspect_precision + aspect_recall)
  else:
    aspect_f1=0
  logger.info('AE - Precision: {}\tRecall: {}\tF1: {}'.format(aspect_precision, aspect_recall, aspect_f1))

  ##Calculate F1-score for opinion:
  opinion_precision = float(opinion_match_num) / float(opinion_predict_num)
  opinion_recall = float(opinion_match_num) / float(opinion_target_num)
  if (opinion_precision + opinion_recall)!=0:
    opinion_f1 = 2 * opinion_precision * opinion_recall / (opinion_precision + opinion_recall)
  else:
    opinion_f1=0
  logger.info('OE - Precision: {}\tRecall: {}\tF1: {}'.format(opinion_precision, opinion_recall, opinion_f1))

  ##Calculate F1-score for sentiment:
  sentiment_precision = float(sentiment_match_num) / float(sentiment_predict_num)
  sentiment_recall = float(sentiment_match_num) / float(sentiment_target_num)
  if (sentiment_precision + sentiment_recall)!=0:
    sentiment_f1 = 2 * sentiment_precision * sentiment_recall / (sentiment_precision + sentiment_recall)
  else:
    sentiment_f1=0
  logger.info('ASC - Precision: {}\tRecall: {}\tF1: {}'.format(sentiment_precision, sentiment_recall, sentiment_f1))

  ##Calculate F1-score for overall:
  overall_precision = float(asp_pol_match_num) / float(asp_pol_predict_num)
  overall_recall = float(asp_pol_match_num) / float(asp_pol_target_num)
  if (overall_precision + overall_recall)!=0:
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
  else:
    overall_f1=0
  logger.info('Overall - Precision: {}\tRecall: {}\tF1: {}'.format(overall_precision, overall_recall, overall_f1))

  return {
      'AE-F1':aspect_f1,
      'OE-F1':opinion_f1,
      'ASC-F1':sentiment_f1,
      'Overall-F1':overall_f1
  }

logger=get_logger('./logs.txt')

def main(args):
  ##Getting path to data
  #logger.info(f"Version: {args.version}")
  data_path=f'{args.data_path}/data_deberta_v3_xsmall.pt'
  standard_data_path = f"{args.data_path}/data_standard.pt"

  ##Loading data
  logger.info(f"Loading data...")
  total_data = torch.load(data_path)
  standard_data = torch.load(standard_data_path)

  ##Splitting data
  train_data=total_data['train']
  dev_data=total_data['dev']
  test_data=total_data['test']
  dev_standard=standard_data['dev']
  test_standard=standard_data['test']

  ##Init model
  logger.info(f"Initial model...")
  model=RFMRC(args)
  if args.ifgpu:
    model = model.cuda()
  
  #print args
  logger.info(args)

  if args.mode=='train':
    args.save_model_path_ae = args.save_model_path + '/' + args.model_type + '/' + args.data_name + '_' + 'best_dev_ae_f1' + '.pth'
    args.save_model_path_oe = args.save_model_path + '/' + args.model_type + '/' + args.data_name + '_' + 'best_dev_oe_f1' + '.pth'
    args.save_model_path_asc = args.save_model_path + '/' + args.model_type + '/' + args.data_name + '_' + 'best_dev_asc_f1' + '.pth'
    args.save_model_path_overall = args.save_model_path + '/' + args.model_type + '/' + args.data_name + '_' + 'best_dev_overall_f1' + '.pth'
    args.save_model_path_test_ae = args.save_model_path + '/' + args.model_type + '/' + args.data_name + '_' + 'best_test_ae_f1' + '.pth'
    args.save_model_path_test_oe = args.save_model_path + '/' + args.model_type + '/' + args.data_name + '_' + 'best_test_oe_f1' + '.pth'
    args.save_model_path_test_asc = args.save_model_path + '/' + args.model_type + '/' + args.data_name + '_' + 'best_test_asc_f1' + '.pth'
    args.save_model_path_test_overall = args.save_model_path + '/' + args.model_type + '/' + args.data_name + '_' + 'best_test_overall_f1' + '.pth'
    batch_num_train = train_data.batch_num_train(args.batch_size)

    # optimizer
    logger.info('initial optimizer......')
    optimizer = Adam(model.parameters(),lr=args.learning_rate)
    # param_optimizer = list(model.named_parameters())
    # optimizer_grouped_parameters = [
    #         {'params': [p for n, p in param_optimizer if "_bert" in n], 'weight_decay': 0.01},
    #         {'params': [p for n, p in param_optimizer if "_bert" not in n],
    #          'lr': args.learning_rate, 'weight_decay': 0.01}]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)

    # load saved model, optimizer and epoch num
    if args.reload and os.path.exists(args.checkpoint_path):
      checkpoint = torch.load(args.checkpoint_path)
      model.load_state_dict(checkpoint['net'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch'] + 1
      logger.info('Reload model and optimizer after training epoch {}'.format(checkpoint['epoch']))
    else:
      start_epoch = 1
      logger.info('New model and optimizer from epoch 0')


    # scheduler
    training_steps = args.epoch_num * batch_num_train
    ##warmup_steps = int(training_steps * args.warm_up)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0, num_training_steps=training_steps)

    # training
    logger.info('begin training......')
    best_dev_ae_f1 = 0
    best_dev_oe_f1 = 0
    best_dev_asc_f1 = 0
    best_dev_overall_f1 = 0

    best_test_ae_f1 = 0
    best_test_oe_f1 = 0
    best_test_asc_f1 = 0
    best_test_overall_f1 = 0
    for epoch in range(start_epoch, args.epoch_num+1):
      model.train()
      model.zero_grad()

      train_dataset = generating_model_dataset(train_data,args.batch_size,shuffle=True,drop_last=False,istrain=True,ifgpu=args.ifgpu)
      
      for batch_index, batch_dict in enumerate(train_dataset):
        optimizer.zero_grad()

        _,_,_,lossA,lossO,lossS=model(batch_dict,model_mode='train')

        ##Calculate loss
        loss_sum=args.alpha*lossA+args.beta*lossO+args.gamma*lossS

        # if 'deberta' in args.model_type:

        #  ##loss_sum = loss_sum.type(torch.cuda.FloatTensor)
        #   ##loss_sum = Variable(loss_sum, requires_grad=True).cuda()'''
        #   loss_sum_2=loss_sum.float()
        #   loss_sum_2=Variable(loss_sum_2,requires_grad=True).cuda()

        #   # loss_sum_2.grad_fn=loss_sum.grad_fn
        #   # loss_sum=loss_sum_2
        #   print(loss_sum_2.grad_fn)
        
        loss_sum.backward()
        optimizer.step()
        scheduler.step()

      ##train_logger  
        if batch_index % 10 == 0:
          logger.info('Epoch:[{}/{}]\t Batch:[{}/{}]\t Loss Sum:{}\t '
                  'Aspect Loss:{};{}\t Opinion Loss:{};{}\t Sentiment Loss:{}'.
                  format(epoch, args.epoch_num, batch_index, batch_num_train,
                          round(loss_sum.item(), 4),
                          round(lossA.item(), 4), round(lossA.item(), 4),
                          round(lossO.item(), 4), round(lossO.item(), 4),
                          round(lossS.item(), 4)))
       
      # validation
      dev_dataset = generating_model_dataset(dev_data,1,shuffle=False,drop_last=False,istrain=False,ifgpu=args.ifgpu)
      f1 = test(model, dev_dataset, dev_standard,logger)
      # save model and optimizer
      if f1['AE-F1'] > best_dev_ae_f1:
        best_dev_ae_f1 = f1['AE-F1']
        logger.info('Model for best AE saved after epoch {}'.format(epoch))
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, args.save_model_path_ae)
      
      if f1['OE-F1'] > best_dev_oe_f1:
        best_dev_oe_f1 = f1['OE-F1']
        logger.info('Model for best OE saved after epoch {}'.format(epoch))
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, args.save_model_path_oe)
      
      if f1['ASC-F1'] > best_dev_asc_f1:
        best_dev_asc_f1 = f1['ASC-F1']
        logger.info('Model for best ASC saved after epoch {}'.format(epoch))
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, args.save_model_path_asc)
      
      if f1['Overall-F1'] > best_dev_overall_f1:
        best_dev_overall_f1 = f1['Overall-F1']
        logger.info('Model for best overall saved after epoch {}'.format(epoch))
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, args.save_model_path_overall)

      # test
      test_dataset = generating_model_dataset(test_data,1,shuffle=False,drop_last=False,istrain=False,ifgpu=args.ifgpu)
      f1 = test(model, test_dataset, test_standard, logger)
      if f1['AE-F1'] > best_test_ae_f1:
        best_test_ae_f1 = f1['AE-F1']
        logger.info('Model for best test AE saved after epoch {}'.format(epoch))
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, args.save_model_path_test_ae)
      
      if f1['OE-F1'] > best_test_oe_f1:
        best_test_oe_f1 = f1['OE-F1']
        logger.info('Model for best test OE saved after epoch {}'.format(epoch))
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, args.save_model_path_test_oe)
      
      if f1['ASC-F1'] > best_test_asc_f1:
        best_test_asc_f1 = f1['ASC-F1']
        logger.info('Model for best test ASC saved after epoch {}'.format(epoch))
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, args.save_model_path_test_asc)
      
      if f1['Overall-F1'] > best_test_overall_f1:
        best_test_overall_f1 = f1['Overall-F1']
        logger.info('Model for best test overall saved after epoch {}'.format(epoch))
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, args.save_model_path_test_overall)
      
      ##Notice about epoch
      logger.info('Epoch {} done!'.format(epoch))
  elif args.mode=='test':
      logger.info('start testing......')
      test_dataset = generating_model_dataset(test_data,1,shuffle=False,drop_last=False,istrain=False,ifgpu=args.ifgpu)
      # load checkpoint
      logger.info('loading checkpoint......')
      checkpoint = torch.load(args.checkpoint_path)
      model.load_state_dict(checkpoint['net'])
      model.eval()
      
      # eval
      logger.info('evaluating......')
      f1 = test(model, test_dataset,test_standard,logger)

        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Role Flipped Machine Reading Comprehension')
    parser.add_argument('--data_path', type=str, default="./data/14resV2/preprocess/")
    parser.add_argument('--log_path', type=str, default="./log/")
    parser.add_argument('--data_name', type=str, default="14res", choices=["14lap", "14res", "15rest", "16rest"])

    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])

    parser.add_argument('--reload', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default="./model/14res/modelFinal.model")
    parser.add_argument('--save_model_path', type=str, default="./model_deberta")
    parser.add_argument('--model_name', type=str, default="1")

    # model hyper-parameter
    parser.add_argument('--model_type', type=str, default="microsoft/deberta-v3-xsmall")
    parser.add_argument('--hidden_size', type=int, default=384)

    # training hyper-parameter
    parser.add_argument('--ifgpu', type=bool, default=True)
    parser.add_argument('--epoch_num', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--p',type=int,default=8)
    parser.add_argument('--q',type=int,default=5)
    parser.add_argument('--T',type=int,default=2)
    parser.add_argument('--lambda_aspect',type=float,default=1)
    parser.add_argument('--lambda_opinion',type=float,default=1)
    parser.add_argument('--alpha',type=float,default=1)
    parser.add_argument('--beta',type=float,default=1)
    parser.add_argument('--gamma',type=float,default=1)

    args = parser.parse_args()

    main(args)