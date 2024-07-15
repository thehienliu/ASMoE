import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, hidden_size, pad_token_id=0, max_position_embeddings=512, layer_norm_eps=1e-12, hidden_dropout_prob=0.1):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids = None,
    ) -> torch.Tensor:

        # Word embedding
        inputs_embeds = self.word_embeddings(input_ids)

        # Position Embeddings
        position_embeddings = self.position_embeddings(self.position_ids[:, :input_ids.shape[1]])

        # Add them together
        embeddings = self.LayerNorm(inputs_embeds + position_embeddings)
        return self.dropout(embeddings)

class MultiHeadedDotAttention(nn.Module):
    def __init__(self, features_size, num_heads = 8, dropout=0.1, causal=False):
        super(MultiHeadedDotAttention, self).__init__()

        assert features_size % num_heads == 0, f'{features_size = } is not divisible by {num_heads = }'
        self.d_model = features_size
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads


        # Create linear projections
        self.query_linear = nn.Linear(features_size, features_size)
        self.key_linear = nn.Linear(features_size, features_size)
        self.value_linear = nn.Linear(features_size, features_size)

        self.causal = causal

        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, att_mask = None):
        scores = torch.matmul(query, key.mT) / math.sqrt(self.d_k) # batch, head, seq_q, seq_k

        if att_mask is not None:
          scores.masked_fill_(att_mask[:, None, None, :] == 0, float('-inf')) # attn_mask.shape[-1] must be equal seq_k

        if self.causal:
          seq_len = query.shape[2] # This Causal mask only apply when using self-attention its mean len_q == len_k
          causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=scores.device))
          scores.masked_fill_(causal_mask[None, None, :, :]==0, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(self.dropout(p_attn), value) # batch, head, seq_q, d_model

    def forward(self, query, key, value, att_mask = None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        query_ = self.query_linear(query) # batch_size, seq_len_q, d_k * num_heads
        key_ = self.key_linear(key)
        value_ = self.value_linear(value)

        # Reshape to batch_size, seq_len, num_heads, d_k
        query_ = query_.reshape(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2) # batch_size, num_heads, seq_len, d_model
        key_ = key_.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        value_ = value_.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        attended = self.attention(query_, key_, value_, att_mask).transpose(1, 2) # batch, head, seq_q, d_model
        return attended.contiguous().reshape(batch_size, seq_len_q, self.num_heads * self.d_k)

class Decoder_Layer(nn.Module):
  def __init__(self, embedding_size, dropout = 0.1):
    super(Decoder_Layer, self).__init__()

    # Self Attention
    self.self_multihead_attn = MultiHeadedDotAttention(embedding_size, causal=True)
    self.self_norm = nn.LayerNorm(embedding_size)

    # Cross Attention
    self.cross_multihead_attn = MultiHeadedDotAttention(embedding_size)
    self.cross_norm = nn.LayerNorm(embedding_size)

    # FFN Layer
    self.mlp = nn.Sequential(
      nn.Linear(embedding_size, embedding_size * 4),
      nn.GELU(),
      nn.Linear(embedding_size * 4, embedding_size),
      nn.Dropout(dropout)
    )

    self.mlp_norm = nn.LayerNorm(embedding_size)

  def forward(self, embedded_answer, answer_mask, cross_feature, cross_maks):

    # Self Attention
    self_answer_features = self.self_norm(embedded_answer + self.self_multihead_attn(embedded_answer,
                                                                                     embedded_answer,
                                                                                     embedded_answer,
                                                                                     answer_mask))

    # Cross Attention
    cross_answer_features = self.cross_norm(self_answer_features + self.cross_multihead_attn(self_answer_features,
                                                                                             cross_feature,
                                                                                             cross_feature,
                                                                                             cross_maks))

    return self.mlp_norm(cross_answer_features + self.mlp(cross_answer_features))

class Decoder(nn.Module):
  def __init__(self, vocab_size, embedding_size, num_layers, embedding_state_dict=None):
    super(Decoder, self).__init__()

    # Load embedding
    self.embedding_layer = BertEmbeddings(vocab_size, embedding_size)
    if embedding_state_dict:
      self.embedding_layer.load_state_dict(embedding_state_dict, strict=False)

    # Decoder and head
    self.decoder_layers = nn.ModuleList([Decoder_Layer(embedding_size) for _ in range(num_layers)])
    self.head = nn.Linear(embedding_size, vocab_size)

  def forward(self, answer_ids, answer_mask, cross_feature, cross_mask):

    embedded_answer = self.embedding_layer(answer_ids)

    for layer in self.decoder_layers:
      embedded_answer = layer(embedded_answer,
                              answer_mask,
                              cross_feature,
                              cross_mask)  # batch_size, sequence_length, 768

    return self.head(embedded_answer) # batch_size, sequence_length, vocab_size

class Basline(nn.Module):
  def __init__(self, vision_module, language_module, crossentropy_output=True):
    super(Basline, self).__init__()

    # Encoder Module
    self.image_encoder = AutoModel.from_pretrained(vision_module) # CVT
    self.question_encoder = AutoModel.from_pretrained(language_module) # mBert
    self.padding_idx = self.question_encoder.config.pad_token_id
    self.crossentropy_output = crossentropy_output

    # Question encoder Size
    embedding_size = self.question_encoder.config.hidden_size
    vocab_size = self.question_encoder.config.vocab_size

    # Image Encoder Size
    self.feature_size = self.image_encoder.config.hidden_size

    # Norm Cross Feature
    self.norm_input = nn.LayerNorm(embedding_size)

    # Decoder Module
    self.answer_decoder = Decoder(vocab_size=vocab_size,
                                  embedding_size=embedding_size,
                                  num_layers=6,
                                  embedding_state_dict=self.question_encoder.embeddings.state_dict())

  def forward(self, samples, device):

    samples['image_'] = samples['image_'].to(device)
    samples['question_'] = samples['question_'].to(device)
    samples['answer_'] = samples['answer_'].to(device)

    # Prepare Input Features
    image_features = self.image_encoder(samples['image_']).last_hidden_state # batch_size, seq_length, hidden_state

    # Image Mask
    image_mask = torch.ones(samples['question_'].attention_mask.shape[0], image_features.shape[1], device=device) # 32, 197

    # Embedding Question
    encoded_question = self.question_encoder(**samples['question_']).last_hidden_state # batch_size, 256, 768

    # Image + Question
    cross_mask = torch.cat([image_mask, samples['question_'].attention_mask], dim = -1) # 
    cross_feature = self.norm_input(torch.cat([image_features, encoded_question], dim = 1)) # batch_size, 1024 + 256, 768


    # Generate Answer
    output = self.answer_decoder(samples['answer_'].input_ids,
                                 samples['answer_'].attention_mask,
                                 cross_feature,
                                 cross_mask)

    # Roll the answer here and
    labels = samples['answer_'].input_ids.roll(shifts=-1, dims=1)
    labels[:, -1] = self.padding_idx

    # tranpose for calculate the crossentropy loss:
    if self.crossentropy_output:
      output = output.transpose(-1, 1)

    return (output, labels)