import numpy as np

# notes on backprop: each layer translates error into its own 'language'
# linked projections of one landscape onto diff var
class Embedding:
    # d_model = size of vectors (dimension of models) - embedding size (columns)
    # vocab_size = # of unique tokens (rows)
    def __init__(self, vocab_size, d_model):
        # prevent exploding activation early in training w/ 0.01
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01

    # forward pass: lookup embeddings -> pass attention/transformer layer -> get predictions
    def forward(self, x):
        return self.embedding[x]

# final_representation = embedding + positional encoding
# learned position embeddings -> works for fixed max length (BERT)
# RoPE (rotary pe) - diff way to use sin/cos, better for longer sequences
# ALiBi (no pe/simpler) - no encoding w/ bias terms in attention
class PositionalEncoding:
    # max_len = longest sentence can process (d_model max per sentence)
    def __init__(self, d_model, max_len=5000):
        self.pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float).reshape(-1, 1)
        # fast waves -> slow waves (prevent collision)
        # nearby = similar encoding ; far = different encoding
        div_term = np.exp(np.arange(0, d_model, 2).reshape(1, -1) * (-np.log(10000.0) / d_model))
        # sin for even indices, cos for odd indices
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)
        self.pe = self.pe.reshape((1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

# (2016) -> mean = 0, std = 1 squasher
class LayerNorm: 
    def __init__(self, d_model):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    # normalization
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        # learnable scale/shift as needed -> representational flexibility
        return self.gamma * (x - mean) / std + self.beta

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model 
        self.num_heads = num_heads
        # Q, K, V
        self.query_linear = np.random.randn(d_model, d_model) * 0.01
        self.key_linear = np.random.randn(d_model, d_model) * 0.01
        self.value_linear = np.random.randn(d_model, d_model) * 0.01
        self.dropout = 0.1

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]

        # learned W_q * Q, W_k * K, W_v * V
        # query = search; key = index; value = content
        # dimension = d_model / num_heads
        # tranpose rearranged heads to process in parallel
        query = np.dot(query, self.query_linear).reshape(batch_size, query_len, self.num_heads, -1).transpose(0, 2, 1, 3)
        key = np.dot(key, self.key_linear).reshape(batch_size, key_len, self.num_heads, -1).transpose(0, 2, 1, 3)
        value = np.dot(value, self.value_linear).reshape(batch_size, key_len, self.num_heads, -1).transpose(0, 2, 1, 3)
        
        # Q * K (similarity) / sqrt(d_k) (normalize variance ~1)
        attention_scores = np.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.d_model / self.num_heads)
        if mask is not None:
            attention_scores += mask

        # softmax
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)
        attention_weights = attention_weights * (1 - self.dropout)

        # weighted sum of values w/ attn weights (weight by importance)
        context = np.matmul(attention_weights, value).transpose(0, 2, 1, 3).reshape(batch_size, query_len, self.d_model)
        return context
    
class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        # previous tokens generated
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # "cross attn" looks at encoder output
        self.encoder_attn = MultiHeadAttention(d_model, num_heads)
        # process info
        self.feed_forward = np.random.randn(d_model, d_ff) * 0.01
        self.feed_forward_output = np.random.randn(d_ff, d_model) * 0.01
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_output=None, self_attn_mask=None, cross_attn_mask=None):
        # self attention
        # x x x = Q K V
        self_attn_output = self.self_attn.forward(x, x, x, self_attn_mask)
        self_attn_output = self.layer_norm1.forward(x + self_attn_output)

        # cross attention
        if encoder_output is not None:
            encoder_attn_output = self.encoder_attn.forward(self_attn_output, encoder_output, encoder_output, encoder_attn_mask)
            encoder_attn_output = self.layer_norm2.forward(self_attn_output + encoder_attn_output)

        # feed forward
        feed_forward_output = np.dot(self_attn_output, self.feed_forward)
        feed_forward_output = np.maximum(feed_forward_output, 0)
        feed_forward_output = np.dot(feed_forward_output, self.feed_forward_output)
        feed_forward_output = self.layer_norm3.forward(self_attn_output + feed_forward_output)
        return feed_forward_output

class Transformer:
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000):
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        # encode input through self-attention + feed-forward
        self.decoder_blocks = [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.final_norm = LayerNorm(d_model)
        self.final_proj = np.random.randn(d_model, vocab_size) * 0.01