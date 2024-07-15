import os
import json
import torch
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset

class EVJVQADataset(Dataset):

  def __init__(self, image_folder, json_files, tokenizer, image_processor):

    with open(json_files) as f:
      data = json.load(f)
      map_dict = {item['id']: item['filename'] for item in data['images']}
      annotations = data['annotations']

    self.data = self.get_data(image_folder, annotations, map_dict)
    self.tokenizer = tokenizer
    self.image_processor = image_processor


  def get_data(self, image_folder, annotations, image_id_to_filename):
    data = []

    for di in tqdm(annotations):
      id, image_id, question, answer = di.values()
      image_path = os.path.join(image_folder, image_id_to_filename[image_id])
      image = Image.open(image_path)
      data.append({'id': id,
                   'image_id': image_id,
                   'image_path': image_path,
                   'question': question,
                   'answer': answer,
                   'image': image})
    return data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    sample = self.data[idx]
    return (sample['id'], 
            sample['image_id'],
            sample['image_path'],
            sample['question'],
            sample['answer'],
            sample['image'])
    
  def __repr__(self) -> str:
    body = f"""Dataset EVJVQADataset
    Number of datapoints: {len(self.data)}
    Item format: id - image_id - image_path - question - answer - image"""
    return body

  def collate_fn(self, batch):
    id, image_id, image_path, question, answer, image = zip(*batch)

    image_preprocess = self.image_processor(image, return_tensors="pt").pixel_values
    question_tokenized = self.tokenizer(question, padding=True, return_tensors='pt')
    answer_tokenized = self.tokenizer(answer, padding=True, return_tensors='pt')

    return {'id': id,
            'image_id': image_id,
            'image_path': image_path,
            'question_r': question,
            'answer_r': answer,
            'pixel_values': image_preprocess,
            'question': question_tokenized,
            'answer': answer_tokenized}