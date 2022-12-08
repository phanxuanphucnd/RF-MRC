###This file curently only use for bert-base-uncased pretrained model


import torch
from model import RFMRC
from transformers import BertTokenizer, DebertaV2Tokenizer
import numpy as np
import argparse

import nltk
nltk.download('punkt')
from nltk import word_tokenize

CLS_id=101
SEP_id=102

DEBERTA_CLS_ID=1
DEBERTA_SEP_ID=2

def sentToInput(sent,tokenizer,args):
    word_list=word_tokenize(sent,language='english')
    sent=' '.join(word_list)
    if 'deberta' in args.model_type:
        text_ids=tokenizer.encode(sent.lower(),add_special_tokens=False)
        initial_question=[DEBERTA_CLS_ID]+[DEBERTA_SEP_ID]+text_ids+[DEBERTA_SEP_ID]
        attention_mask=[1]*len(initial_question)
        token_type_ids=[0]*2+[1]*(len(initial_question)-2)
        opinion_answers=[-1]*len(initial_question)
        aspect_answers=[-1]*len(initial_question)
        sentiments=[-1]*len(initial_question)
        ignore_index=[]
        temp_text_ids=[]
        for tok in word_list:
            ids=tokenizer.encode(tok.lower(),add_special_tokens=False)
            temp_text_ids+=ids
            ignore_index.append(0)
            for _ in range(len(ids[1:])):
                ignore_index.append(-1)

        assert temp_text_ids==text_ids ##Đảm bảo giữa phần encode từng từ và encode nguyên câu là giống nhau
    else:
        temp_text=tokenizer.tokenize(sent)
        text_ids=tokenizer.convert_tokens_to_ids([word.lower() for word in temp_text])
        initial_question=[CLS_id]+[SEP_id]+text_ids+[SEP_id]
        attention_mask=[1]*len(initial_question)
        token_type_ids=[0]*2+[1]*(len(initial_question)-2)
        opinion_answers=[-1]*len(initial_question)
        aspect_answers=[-1]*len(initial_question)
        sentiments=[-1]*len(initial_question)
        ignore_index=[0]*len(initial_question)
    if args.ifgpu==True:
        return {
            'texts':[word_list],
            'texts_ids': [text_ids],
            'initial_input_ids': torch.tensor([initial_question]).cuda(),
            'initial_attention_mask': torch.tensor([attention_mask]).cuda(),
            'initial_token_type_ids': torch.tensor([token_type_ids]).cuda(),
            'aspect_answers':torch.tensor([aspect_answers]).cuda(),
            'opinion_answers':torch.tensor([opinion_answers]).cuda(),
            'sentiments':torch.tensor([sentiments]).cuda(),
            'ignore_indexes':[ignore_index]
        }
    else:
        return {
            'texts':[word_list],
            'texts_ids': [text_ids],
            'initial_input_ids': torch.tensor([initial_question]),
            'initial_attention_mask': torch.tensor([attention_mask]),
            'initial_token_type_ids': torch.tensor([token_type_ids]),
            'aspect_answers':torch.tensor([aspect_answers]),
            'opinion_answers':torch.tensor([opinion_answers]),
            'sentiments':torch.tensor([sentiments]),
            'ignore_indexes':[ignore_index]
        }


def indexToToken(index_list,tokens):
    result=[]
    for indexes in index_list:
        for index in indexes:
            text=''
            for i in range(index[0],index[1]+1):
                text+=tokens[i]+' '
            result.append(text[:-1])
    return result

def inference(args):
    sentence=args.sentence
    model=RFMRC(args)
    if args.ifgpu:
        model = model.cuda()
    checkpoint=torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    if 'deberta' in args.model_type:
        _tokenizer=DebertaV2Tokenizer.from_pretrained(args.model_type)
    else:
        _tokenizer=BertTokenizer.from_pretrained(args.model_type)
    model_input=sentToInput(sentence,_tokenizer,args)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))

    #GPU-WARM-UP
    for _ in range(10):
        _,_,_,_,_,_ = model(model_input,model_mode=args.mode)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            aspect_terms,opinion_terms,sentiments,_,_,_=model(model_input,model_mode=args.mode)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    
    id_to_sentiment={0:'POS',1:'NEG',2:'NEU',-1:None}
    tokens=word_tokenize(sentence,language='english')
    aspects=indexToToken(aspect_terms,tokens)
    opinions=indexToToken(opinion_terms,tokens)
    asp_pol=[]
    for i in range(len(aspect_terms[0])):
        asp=aspect_terms[0][i]
        asp_pol.append((aspects[i],id_to_sentiment[sentiments[0][asp[0]]]))

    
    return asp_pol,opinions,mean_syn

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Inference')
    ##Define path where save unprocessed data and where to save processed data
    parser = argparse.ArgumentParser(description='Role Flipped Machine Reading Comprehension')
    parser.add_argument('--sentence', type=str, default="Great mains, pity about the chips.")

    parser.add_argument('--mode', type=str, default="test", choices=["train", "test"])

    parser.add_argument('--checkpoint_path', type=str, default="./best_model/14res_best_test_overall_f1_deberta_v3_xsmall.pth")

    # model hyper-parameter
    parser.add_argument('--model_type', type=str, default="microsoft/deberta-v3-xsmall")
    parser.add_argument('--hidden_size', type=int, default=384)

    # training hyper-parameter
    parser.add_argument('--ifgpu', type=bool, default=False)
    parser.add_argument('--p',type=int,default=8)
    parser.add_argument('--q',type=int,default=5)
    parser.add_argument('--T',type=int,default=2)

    args = parser.parse_args()

    asp_pol,opinions,time=inference(args)

    print('Sentence:',args.sentence)
    print('Aspect and polarity:',asp_pol)
    print('Opinion terms',opinions)
    print('Inference time',time)