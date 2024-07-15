import evaluate
import numpy as np

meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")

def convert_ids_to_string(ids, tokenizer):

  new_ids = ids.detach().clone()
  new_strings = []

  if new_ids.dim() == 3:
    new_ids = new_ids.argmax(-1) # batch_size, seq_len

  return [tokenizer.decode(id, skip_special_tokens=True) for id in new_ids]

def compute_accuracy(outputs, labels, padding_idx):

  flat_labels = labels.reshape(-1).detach().clone() # batch_size * seq_len
  flat_outputs = outputs.reshape(-1).detach().clone() # batch_size * seq_len

  mask = flat_labels != padding_idx

  flat_labels = flat_labels[mask]
  flat_outputs = flat_outputs[mask]

  acc = (flat_outputs == flat_labels).sum().item() / flat_labels.shape[0]
  return acc

def compute_metrics(outputs, label_ids, tokenizer):

  output_ids = outputs.detach().clone().argmax(dim=-1) # batch_size, seq_len

  pred_ans = convert_ids_to_string(output_ids, tokenizer)
  ground_t = convert_ids_to_string(label_ids, tokenizer)
    
  if pred_ans[0] == '':
    pred_ans[0] += '<avoid_float_division_by_zero>'

  # Compute BLEU, ROUGE, METEOR
  accuracy_score = compute_accuracy(output_ids, label_ids, tokenizer.pad_token_id)
  bleu1_score = bleu.compute(predictions=pred_ans, references=ground_t, max_order=1)['bleu']
  bleu2_score = bleu.compute(predictions=pred_ans, references=ground_t, max_order=2)['bleu']
  bleu3_score = bleu.compute(predictions=pred_ans, references=ground_t, max_order=3)['bleu']
  bleu4_score = bleu.compute(predictions=pred_ans, references=ground_t, max_order=4)['bleu']
  rouge_score = rouge.compute(predictions=pred_ans, references=ground_t)['rougeL']
  meteor_score = meteor.compute(predictions=pred_ans, references=ground_t)['meteor']

  return np.array([accuracy_score, bleu1_score, bleu2_score, bleu3_score, bleu4_score, rouge_score, meteor_score])