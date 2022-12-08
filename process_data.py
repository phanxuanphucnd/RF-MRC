import re
import argparse
from tqdm import tqdm
from dataset import BasicSample
import os
import torch

def get_text(lines):
  text_list=[]
  triples=[]
  sent_to_id={'POS':0,'NEG':1,'NEU':2}
  for line in lines:
    triplet_list=[]
    text,triplets=line.split('####')
    word_list=text.split()
    text_list.append(word_list)
    ##Processing trilets string to triplet data
    triplets_string=triplets[1:-1]
    ##Finding start and end of each triplet
    start=[m.start() for m in re.finditer('\((.*?)\)', triplets_string)]
    end=[m.end() for m in re.finditer('\((.*?)\)', triplets_string)]
    ##Iterate through each triplet
    for i in range(len(start)):
      aspect_string,opinion_string,sentiment_string=triplets_string[start[i]+1:end[i]-1].split('], ')
      ##Iterate through each element to process convert to expected data
      ##Aspect
      aspect_list=[]
      aspect_string_list=aspect_string[1:].split(', ')
      for aspect_string in aspect_string_list:
        aspect_list.append(int(aspect_string))

      ##Opinion
      opinion_list=[]
      opinion_string_list=opinion_string[1:].split(', ')
      for opinion_string in opinion_string_list:
        opinion_list.append(int(opinion_string))
      
      ###Sentiment
      sentiment=sent_to_id[sentiment_string[1:-1]]
      triplet_list.append((aspect_list,opinion_list,sentiment))
    triples.append(triplet_list)
  return text_list,triples

def fusion_triplet(triplet):
  triplet_aspect=[]
  triplet_opinion=[]
  triplet_sentiment=[]
  for t in triplet:
    if t[0] not in triplet_aspect:
      triplet_aspect.append(t[0])
    if t[1] not in triplet_opinion:
      triplet_opinion.append(t[1])
    triplet_sentiment.append(t[2])
  return triplet_aspect,triplet_opinion,triplet_sentiment

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Processing data')
    ##Define path where save unprocessed data and where to save processed data
    parser.add_argument('--data_path', type=str, default="./data/14resV2")
    parser.add_argument('--output_path', type=str, default="./data/14resV2/preprocess")
    
    args=parser.parse_args()
    ##Begin processing flow
    DATASET_TYPE_LIST=['train','dev','test']

    for dataset_type in DATASET_TYPE_LIST:
    ##Reading the triple span labelling data
    ##File pkl contain pair of labeling start and end of aspect terms and opinion terms and sentiment polarity label
  
        ##Reading the text where these label are labelled
        with open(f'{args.data_path}/{dataset_type}_triplets.txt','r',encoding='utf-8') as f:
            text_lines=f.readlines()

        ##Getting the text
        text_list,triple_data=get_text(text_lines)

        ##Initial sample list
        sample_list=[]
        header_fmt='Processing {:>5s}'
        for i in tqdm(range(len(text_list)),desc=f'{header_fmt.format(dataset_type.upper())}'):
            triplet=triple_data[i] ##Get one triplet in triple label data
            text=text_list[i]  ##Getting one text in all text data

            ##Creating list of start end pos aspect opinion
            triplet_aspect,triplet_opinion,triplet_sentiment=fusion_triplet(triplet)

            ##Define some list to save data for training
            aspect_query_list=[]
            opinion_answer_list=[]
            opinion_query_list=[]
            aspect_answer_list=[]

            aspect_label=[0]*len(text)
            opinion_label=[0]*len(text)
            ##Test if NULL not -1 but 3
            sentiment_label=[-1]*len(text)
            aspect_as_query=''
            opinion_as_query=''
            for j in range(len(triplet_aspect)):
                ta=triplet_aspect[j]
                s=triplet_sentiment[j]
                sentiment_label[ta[0]:ta[-1]+1]=[s]*len(text[ta[0]:ta[-1]+1])
                aspect_label[ta[0]]=1
                aspect_label[ta[0]+1:ta[-1]+1]=[2]*len(text[ta[0]+1:ta[-1]+1])
                aspect_as_query=aspect_as_query+' '.join(text[ta[0]:ta[-1]+1])+' '
            aspect_as_query=aspect_as_query[:-1].split()
            for to in triplet_opinion:
                opinion_label[to[0]]=1
                opinion_label[to[0]+1:to[-1]+1]=[2]*len(text[to[0]+1:to[-1]+1])
                opinion_as_query=opinion_as_query+' '.join(text[to[0]:to[-1]+1])+' '
            opinion_as_query=opinion_as_query[:-1].split()
            aspect_query_list.append(aspect_as_query) 
            opinion_query_list.append(opinion_as_query)
            opinion_answer_list.append(opinion_label)
            aspect_answer_list.append(aspect_label)
    
            ##Creating ProcessedSample and save to sample_list
            sample = BasicSample(
                text,
                aspect_query_list[0],
                opinion_answer_list[0],
                opinion_query_list[0],
                aspect_answer_list[0],
                sentiment_label
            )
            sample_list.append(sample)

        ##Save the processed data
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        output_path=f'{args.output_path}/{dataset_type}_PREPROCESSED.pt'
        print(f"Saved data to `{output_path}`.")
        torch.save(sample_list,output_path)