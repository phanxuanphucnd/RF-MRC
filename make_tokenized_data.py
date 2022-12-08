from transformers import BertTokenizer,DebertaV2Tokenizer
from tqdm import tqdm
from dataset import ProcessedIdDataset
import argparse
import torch
import os

def tokenized_and_process(data, args,mode='train'):
  if 'deberta' in args.model_type:
      _tokenizer=DebertaV2Tokenizer.from_pretrained(args.model_type)
  else:
      _tokenizer=BertTokenizer.from_pretrained(args.model_type)

  text_list=[]
  text_id_list=[]

  aspect_question_list=[]
  aspect_question_id_list=[]
  opinion_answer_list=[]

  opinion_question_list=[]
  opinion_question_id_list=[]
  aspect_answer_list=[]
  ignore_indexes=[]

  sentiment_list=[]
  header_fmt='Tokenize data {:>5s}'
  for sample in tqdm(data,desc=f'{header_fmt.format(mode.upper())}'):

    ##Temp data
    ###Text
    temp_text=sample.text_tokens
    text_list.append(temp_text)
    ##Aspect question
    aspect_question=sample.aspect_queries
    aspect_question_list.append(aspect_question)
    ##Opinion question
    opinion_question=sample.opinion_queries
    opinion_question_list.append(opinion_question)

    ignore_index=[]

    ###Convert tokens to ids

    
    if 'deberta' not in args.model_type:
        ##The text
        text_ids=_tokenizer.convert_tokens_to_ids(
            [word.lower() for word in temp_text]
        )
        text_id_list.append(text_ids)

        ##Aspect question
        aspect_queries_ids=_tokenizer.convert_tokens_to_ids(
            [word.lower() for word in aspect_question]
        )
        aspect_question_id_list.append(aspect_queries_ids)

        ##Opinion question
        opinion_queries_ids=_tokenizer.convert_tokens_to_ids(
            [word.lower() for word in opinion_question]
        )
        opinion_question_id_list.append(opinion_queries_ids)

        assert len(text_ids)==len(sample.aspect_answers)==len(sample.opinion_answers)

        #Apsect answer
        aspect_answer_list.append(sample.aspect_answers)

        #Opinion answer
        opinion_answer_list.append(sample.opinion_answers)

        #Sentiment
        sentiment_list.append(sample.sentiments)
    else:
        aspect_answer=[]
        opinion_answer=[]
        sentiment=[]
        ignore_index=[]

        ##The text
        text_ids=_tokenizer.encode(' '.join(temp_text).lower(),add_special_tokens=False)
        text_id_list.append(text_ids)

        ##Aspect question
        aspect_queries_ids=_tokenizer.encode(' '.join(aspect_question).lower(),add_special_tokens=False)
        aspect_question_id_list.append(aspect_queries_ids)

        ##Opinion question
        opinion_queries_ids=_tokenizer.encode(' '.join(opinion_question).lower(),add_special_tokens=False)
        opinion_question_id_list.append(opinion_queries_ids)

        ##Điều chỉnh lại nhãn cho phù hợp với encode của deberta
        temp_text_ids=[]
        for ind,tok in enumerate(temp_text):
            ids=_tokenizer.encode(tok.lower(),add_special_tokens=False)
            temp_text_ids+=ids
            aspect_answer.append(sample.aspect_answers[ind])
            opinion_answer.append(sample.opinion_answers[ind])
            sentiment.append(sample.sentiments[ind])
            ignore_index.append(0)
            for _ in range(len(ids[1:])):
                ignore_index.append(-1)
                aspect_answer.append(-1)
                opinion_answer.append(-1)
                sentiment.append(-1)


        assert temp_text_ids==text_ids ##Đảm bảo giữa phần encode từng từ và encode nguyên câu là giống nhau
        assert len(ignore_index)==len(aspect_answer)==len(opinion_answer)==len(sentiment)==len(text_ids) ###Đảm bảo phần nhãn nằm đúng ở các vị trí
                
        #Apsect answer
        aspect_answer_list.append(aspect_answer)

        #Opinion answer
        opinion_answer_list.append(opinion_answer)

        #Sentiment
        sentiment_list.append(sentiment)

        ##Ignore_indexes
        ignore_indexes.append(ignore_index)

  result={
      'texts':text_list,
      'texts_ids':text_id_list,
      'aspect_questions':aspect_question_list,
      'aspect_questions_ids':aspect_question_id_list,
      'opinion_answers':opinion_answer_list,
      'opinion_questions':opinion_question_list,
      'opinion_questions_ids':opinion_question_id_list,
      'aspect_answers':aspect_answer_list,
      'sentiments':sentiment_list,
      'ignore_indexes':ignore_indexes
  }

  final_data=ProcessedIdDataset(result)
  return final_data

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Processing data')
    ##Define path where save unprocessed data and where to save processed data
    parser.add_argument('--data_path', type=str, default="./data/14resV2/preprocess")
    parser.add_argument('--output_path', type=str, default="./data/14resV2/preprocess")
    parser.add_argument('--model_type',type=str,default='microsoft/deberta-v3-xsmall')
    
    args=parser.parse_args()

    train_data_path = f"{args.data_path}/train_PREPROCESSED.pt"
    dev_data_path = f"{args.data_path}/dev_PREPROCESSED.pt"
    test_data_path = f"{args.data_path}/test_PREPROCESSED.pt"

    train_data=torch.load(train_data_path)
    dev_data=torch.load(dev_data_path)
    test_data=torch.load(test_data_path)

    '''##Making tokenize data before preprocess to id
    train_tokenized,train_max_len=tokenize_data(train_data,version=args_version,mode='train')
    dev_tokenized,dev_max_len=tokenize_data(dev_data,version=args_version,mode='dev')
    test_tokenized,test_max_len=tokenize_data(test_data,version=args_version,mode='test')'''

    '''print(f"train_max_len : {train_max_len}")
    print(f"dev_max_len : {dev_max_len}")
    print(f"test_max_len : {test_max_len}")'''

    ##Processing tokenied data to ids
    train_preprocess = tokenized_and_process(train_data,args,mode='train')
    dev_preprocess = tokenized_and_process(dev_data, args,mode='dev')
    test_preprocess = tokenized_and_process(test_data, args,mode='test')

    ##Saving preprocessing full data
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    output_path=f'{args.output_path}/data_deberta_v3_xsmall.pt'
    print(f"Saved data : `{output_path}`.")
    torch.save({
        'train':train_preprocess,
        'dev':dev_preprocess,
        'test':test_preprocess
    },output_path)