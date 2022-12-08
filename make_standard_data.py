import argparse
from process_data import get_text
from tqdm import tqdm
import os
import torch

##This cell contain make_standard function to make standard data
def make_standard(data_path,dataset_type):
  ##Reading label data
  with open(f'{data_path}/{dataset_type}_triplets.txt','r',encoding='utf-8') as f:
    text_lines=f.readlines()

  _,triple_data=get_text(text_lines)
  
  standard_list=[] ##To save standard data

  header_fmt='Make standard {:>4s}'
  for triplet in tqdm(triple_data,desc=f'{header_fmt.format(dataset_type.upper())}'):
    aspect_temp=[]
    opinion_temp=[]
    pair_temp=[]
    triplet_temp=[]
    asp_pol_temp=[]
    for temp_t in triplet:
      triplet_temp.append([temp_t[0][0],temp_t[0][-1],temp_t[1][0],temp_t[1][-1],temp_t[2]])
      ap=[temp_t[0][0],temp_t[0][-1],temp_t[2]]
      if ap not in asp_pol_temp:
        asp_pol_temp.append(ap)
      a=[temp_t[0][0],temp_t[0][-1]]
      if a not in aspect_temp:
        aspect_temp.append(a)
      o=[temp_t[1][0],temp_t[1][-1]]
      if o not in opinion_temp:
        opinion_temp.append(o)
      p=[temp_t[0][0],temp_t[0][-1],temp_t[1][0],temp_t[1][-1]]
      if p not in pair_temp:
        pair_temp.append(p)

    standard_list.append({
        'asp_target':aspect_temp,
        'opi_target':opinion_temp,
        'asp_opi_target':pair_temp,
        'asp_pol_target':asp_pol_temp,
        'triplet':triplet_temp
    })
  return standard_list

if __name__=='__main__':
    ##Make standard data to evaluate and compare the result
    parser = argparse.ArgumentParser(description='Making standard data')
    ##Define path where save unprocessed data and where to save processed data
    parser.add_argument('--data_path', type=str, default="./data/14resV2")
    parser.add_argument('--output_path', type=str, default="./data/14resV2/preprocess")

    args=parser.parse_args()

    dev_standard=make_standard(args.data_path,'dev')
    test_standard=make_standard(args.data_path,'test')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    output_path=f'{args.output_path}/data_standard.pt'

    print(f"Saved data : `{output_path}`.")
    torch.save({
        'dev':dev_standard,
        'test':test_standard
    },output_path)