from torch.utils.data import Dataset, DataLoader

##First Sample
class BasicSample(object):
  '''
    - Kiểu dữ kiệu chứa những thông tin cơ bản của dữ liệu train, test dev:
      + Token của câu input
      + aspect_query: các token aspect được dùng làm query
      + opinion_query: các token opinion được dùng làm query
      + aspect_answer: các token aspect làm câu trả lời đưới dạng chuỗi nhãn {0:O; B:1, I:2}
      + opinion_answer: các token opinion làm câu trả lời đưới dạng chuỗi nhãn {0:O; B:1, I:2}
      + sentiment: các nhãn cảm xúc cho từng aspect term (những term ngoài aspect được gán là -1)
  '''
  def __init__(
      self,
      text_tokens:str=None,
      aspect_queries:list=None,
      opinion_answers:list=None,
      opinion_queries:list=None,
      aspect_answers:list=None,
      sentiments:list=None
  ):
    self.text_tokens=text_tokens
    self.aspect_queries=aspect_queries
    self.opinion_answers=opinion_answers
    self.opinion_queries=opinion_queries
    self.aspect_answers=aspect_answers
    self.sentiments=sentiments
  
  def printSample(self):
    ##In ra sample hiện tại
    print('Information of this sample:')
    print(f'Text tokens: {self.text_tokens}')
    print(f'Aspect queries: {self.aspect_queries}')
    print(f'Opinion answers: {self.opinion_answers}')
    print(f'Opinion queries: {self.opinion_queries}')
    print(f'Aspect answers: {self.aspect_answers}')
    print(f'Sentiments: {self.sentiments}')

##Id Dataset
class ProcessedIdDataset(Dataset):
  '''
    Một dạng dataset chứa thêm những thông tin đã chuyển từ token sang ID của các câu input
      Có thêm thông tin như: id của text, của aspect query và của opinion query
  '''
  def __init__(self,pre_data):
    self.texts=pre_data.get('texts',None)
    self.texts_ids=pre_data.get('texts_ids',None)
    self.aspect_questions=pre_data.get('aspect_questions',None)
    self.aspect_questions_ids=pre_data.get('aspect_questions_ids',None)
    self.opinion_answers=pre_data.get('opinion_answers',None)
    self.opinion_questions=pre_data.get('opinion_questions',None)
    self.opinion_questions_ids=pre_data.get('opinion_questions_ids',None)
    self.aspect_answers=pre_data.get('aspect_answers',None)
    self.sentiments=pre_data.get('sentiments',None)
    self.ignore_indexes=pre_data.get('ignore_indexes',None)

  def printIndexSample(self,idx):
    print(f'Information of number {idx} sample:')
    print(f'Text: {self.texts[idx]}')
    print(f'Text ids: {self.texts_ids[idx]}')
    print(f'Aspect question: {self.aspect_questions[idx]}')
    print(f'Aspect question ids: {self.aspect_questions_ids[idx]}')
    print(f'Opinion answer: {self.opinion_answers[idx]}')
    print(f'Opinion question: {self.opinion_questions[idx]}')
    print(f'Opinion question ids: {self.opinion_questions_ids[idx]}')
    print(f'Aspect answer: {self.aspect_answers[idx]}')
    print(f'Sentimets: {self.sentiments[idx]}')
    print(f'Ignore index: {self.ignore_indexes[idx]}')

  def __len__(self):
    return len(self.texts)

  def batch_num_train(self,batch_size):
    return len(self.texts)//batch_size
  
  def getIndexSample(self,idx):
    return {
        'text':self.texts[idx],
        'text_ids':self.texts_ids[idx],
        'aspect_question':self.aspect_questions[idx],
        'aspect_question_ids':self.aspect_questions_ids[idx],
        'opinion_answer':self.opinion_answers[idx],
        'opinion_question':self.opinion_questions[idx],
        'opinion_question_ids':self.opinion_questions_ids[idx],
        'aspect_answer':self.aspect_answers[idx],
        'sentiment':self.sentiments[idx],
        'ignore_index':self.ignore_indexes[idx]
    }