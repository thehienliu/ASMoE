import torch
from evaluate import load
from typing import Callable
from sklearn.metrics import accuracy_score, f1_score
from transformers.tokenization_utils_base import BatchEncoding

def flatten_list(li):
  return [i for l in li for i in l]

def indexing(li, idxs):
  return [li[idx] for idx in idxs]

class ComputeMetrics:

  def __init__(self, tokenizer: Callable, return_example_prediction: bool=True):

    self.tokenizer = tokenizer
    self.return_example_prediction = return_example_prediction
    self.meteor = load('meteor')
    self.rouge = load('rouge')
    self.bleu = load("bleu")

  def compute(self, outputs: torch.Tensor, answers_tokenized: BatchEncoding):
    """
    Args:
        outputs (`torch.Tensor` with shape `batch_size, seq_len, vocab_size`): Batch predictions
        answers_tokenized (`BatchEncoding`): Tokenized answers that contains input_ids and attention_mask

    Returns:
        Dict[str, float]: a dictionary with metrics score.
    """

    # Get id, string output
    targets, predicts = [], []

    for i, m in enumerate(answers_tokenized.attention_mask.bool().tolist()):
      targets.append(answers_tokenized.input_ids[i][m].tolist())
      predicts.append(outputs[i][m].tolist())

    # Convert to strings
    references = [self.tokenizer.decode(id) for id in targets]
    predictions = [self.tokenizer.decode(id) for id in predicts]

    # Check empty
    predictions = [p if p.strip() else '<empty>' for p in predictions]

    # Flatten
    targets = flatten_list(targets)
    predicts = flatten_list(predicts)

    # Compute metrics
    accuracy = accuracy_score(targets, predicts)
    f1 = f1_score(targets, predicts, average='micro')
    bleu1_score = self.bleu.compute(predictions=predictions, references=references, max_order=1)['bleu']
    bleu2_score = self.bleu.compute(predictions=predictions, references=references, max_order=2)['bleu']
    bleu3_score = self.bleu.compute(predictions=predictions, references=references, max_order=3)['bleu']
    bleu4_score = self.bleu.compute(predictions=predictions, references=references, max_order=4)['bleu']
    rouge_score = self.rouge.compute(predictions=predictions, references=references)['rougeL']
    meteor_score = self.meteor.compute(predictions=predictions, references=references)['meteor']

    scores = {
        'accuracy': accuracy,
        'f1': f1,
        'bleu1': bleu1_score,
        'bleu2': bleu2_score,
        'bleu3': bleu3_score,
        'bleu4': bleu4_score,
        'rouge': rouge_score,
        'meteor': meteor_score,
    }
    if self.return_example_prediction:
      rand_idx = torch.randint(0, len(references), size=[5 if len(references) > 5 else len(references)])
      gts = indexing(references, rand_idx)
      preds = indexing(predictions, rand_idx)
      scores = (scores, {'ground-truth': gts, 'predictions': preds})

    return scores