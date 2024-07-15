import os
import json
import torch
from tqdm.auto import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Callable, Dict, List, Union
from transformers.tokenization_utils_base import BatchEncoding

class EVJVQAOutput:

    """
    Format item in EVJVQAOutput that contains question-answer pairs, image, and metadata of that item.

    Args:
        image_ids (`Tuple[int]`): Tuple of unique integers for each image. One image can be replicated in many question-answer pairs.
        question_answer_ids (`Tuple[int]`): Tuple of unique integers for each question-answer pair.
        questions (`Tuple[str]`): Tuple of original questions as strings.
        answers (`Tuple[str]`): Tuple of original answers as strings.
        image_features (`torch.FloatTensor` of shape `(1, num_patches, hidden_size)`): The image features after passing through a pretrained model.
        question_features (`torch.FloatTensor` of shape `(1, sequence_length, hidden_size)`): The question features after passing through a pretrained model.
        answers_tokenized: Tokenized answers using a specific tokenizer from the transformers library.
        questions_tokenized: Tokenized questions using a specific tokenizer from the transformers library.
    """  

    def __init__(self,
                 image_ids: Tuple[int],
                 question_answer_ids: Tuple[int],
                 questions: Tuple[str],
                 answers: Tuple[str],
                 image_features: torch.FloatTensor,
                 question_features: torch.FloatTensor,
                 answers_tokenized: BatchEncoding,
                 questions_tokenized: BatchEncoding):
      
      self.data = {
          'image_ids': image_ids,
          'question_answer_ids': question_answer_ids,
          'questions': questions,
          'answers': answers,
          'image_features': image_features,
          'question_features': question_features,
          'answers_tokenized': answers_tokenized,
          'questions_tokenized': questions_tokenized,
      }

    def __getitem__(self, item: str):
        return self.data[item]
    
    def __getattr__(self, item: str):
        return self.data[item]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __repr__(self) -> str:

      questions_text = "\n".join("\t" + str(i) for i in self.data['questions'])
      answers_text = "\n".join("\t" + str(i) for i in self.data['answers'])

      body = [
          f"image_ids: \n{self.data['image_ids']}\n",
          f"question_answer_ids: \n{self.data['question_answer_ids']}\n",
          f"questions: \n{questions_text}\n",
          f"answers: \n{answers_text}\n",
          f"image_features: \n{self.data['image_features']}\n",
          f"question_features: \n{self.data['question_features']}\n",
          f"answers_tokenized: \n{self.data['answers_tokenized']}\n",
          f"questions_tokenized: \n{self.data['questions_tokenized']}\n"]
      return '\n'.join(body)

    def to(self, device: Union[str, "torch.device"]):
      # Bring only tensor data to device
      self.data['image_features'] = self.data['image_features'].to(device=device)
      self.data['question_features'] = self.data['question_features'].to(device=device)
      self.data['answers_tokenized'] = self.data['answers_tokenized'].to(device=device)
      self.data['questions_tokenized'] = self.data['questions_tokenized'].to(device=device)
      return self


class DecodeEVJVQADataset(Dataset):
  def __init__(self,
               json_file: str,
               images_dir: str,
               image_features_dir: str,
               question_features_dir: str,
               tokenizer: Callable):
    """
    Dataset module for traininng decoder using evjvqa dataset.
    Link challenge and dataset: https://vlsp.org.vn/vlsp2022/eval/evjvqa

    Args:
        json_file (`str`): The path of json file that contains images and annotations information.
        image_dir (`str`): The path of image directory that contains images for training model.
        image_features_dir (`str`): The path of image feature that extract from pretrained vision model.
        question_features_dir (`str`): The path of question feature that extract from pretrained language model.
        tokenizer (`Callable`): The callable function or class from transformers to tokenize the answer.
    """
    # Setup for data loading
    data = json.load(open(json_file))
    map_dict = {item['id']: item['filename'] for item in data['images']}
    annotations = data['annotations']

    self.tokenizer = tokenizer
    self.images_dir = images_dir
    self.image_features_dir = image_features_dir
    self.question_features_dir = question_features_dir
    self.data = self.get_data(annotations, map_dict)


  def get_data(self, annotations: List[Dict], image_id_to_filename: Dict) -> List[Dict]:
    data = []

    for di in tqdm(annotations):
      
      id, image_id, question, answer = di.values()
      image_name = image_id_to_filename[image_id]
      image_path = os.path.join(self.images_dir, image_name)
      image_feature_path = os.path.join(self.image_features_dir, f"{image_name.split('.')[0]}.pt")
      question_feature_path = os.path.join(self.question_features_dir, f'{id}.pt')

      data.append({'image_id': image_id,
                   'question_answer_id': id,
                   'image_path': image_path,
                   'question': question,
                   'answer': answer,
                   'image_feature_path': image_feature_path,
                   'question_feature_path': question_feature_path})
    return data

  def __repr__(self) -> str:
    body = f"""Dataset {self.__class__.__name__}
    Image directory: {self.images_dir}
    Image feature directory: {self.image_features_dir}
    Question directory: {self.question_features_dir}
    Tokenizer: {self.tokenizer.__class__.__name__}
    Number of datapoints: {len(self.data)}"""
    return body

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, idx):

    sample = self.data[idx]
    return (sample['image_id'],
            sample['question_answer_id'],
            sample['question'],
            sample['answer'],
            torch.load(sample['image_feature_path']),
            torch.load(sample['question_feature_path'])[0])

  def collate_fn(self, batch):
    image_ids, question_answer_ids, questions, answers, image_features, question_features = zip(*batch)
    image_features = torch.cat(image_features, dim=0)
    question_features = pad_sequence(question_features, batch_first=True)
    questions_tokenized = self.tokenizer(questions, padding=True, return_tensors='pt')
    answers_tokenized = self.tokenizer(answers, padding=True, return_tensors='pt')

    return EVJVQAOutput(image_ids=image_ids,
                        question_answer_ids=question_answer_ids,
                        questions=questions,
                        answers=answers,
                        image_features=image_features,
                        question_features=question_features,
                        answers_tokenized=answers_tokenized,
                        questions_tokenized=questions_tokenized)