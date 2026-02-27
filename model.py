import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

# model = InputEmbedding(512, 1000)
# text = ['My', 'name', 'is', 'Om']
# unique_words = list(set(text))
# vocab = {word: idx for idx, word in enumerate(unique_words)}
# ids = [vocab[word] for word in text]
# print(ids)
# text = torch.tensor(ids, dtype=torch.long)
# output = model(text)
# print("IDs:", ids)
# print(output)
# print("Output shape:", output.shape)


class PositionEncoding(nn.Module):
    def __init__(self, seq_len, d_model, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # (seq_len, 1)
        position = torch.arange(0, seq_len).unsqueeze(1)

        # (d_model/2,)
        division = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * division)
        pe[:, 1::2] = torch.cos(position * division)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
    

# class PositionEncoding(nn.Module):
#     def __init__(self, seq_len, d_model, dropout):
#         super().__init__()

#         self.seq_len = seq_len
#         self.d_model = d_model
#         self.dropout = nn.Dropout(dropout)
        
#         # Create position encoding matrix
#         position = torch.zeros(seq_len, d_model) # [seq_lem, d_model]

#         position = torch.arange(0, seq_len, dtype=torch.float)
#         division = torch.exp( torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
#         # e(- (i/d_model) * log(1000))

#         # Apply sin and cos
#         pe[:, 0::2] = torch.sin(position * division)
#         pe[:, 1::2] = torch.cos(position * division)
#         position = torch.arange(0, seq_len).unsqueeze(1)  # (seq_len, 1)
#         pe = pe.unsqueeze(0)

#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         # x.shape- > (batch_size, seq_len, d_model)

#         # :x.shape[1] -> take only required sequence length
#         x = x + (self.pe[:, : x.shape[1], :]).requires_grad(False)

#         return self.dropout(x)
      


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)   # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)   # W2 and b2

    def forward(self, x):
        # (Batch, Seq_Len, d_model) 
        # -> (Batch, Seq_Len, d_ff) 
        # -> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.d_k = d_model // h
        self.h = h

    # By using staticmethod -> we can call like this function.add(2,3)
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, seq, d_k) ->  (Batch, h, seq, seq)
        # Calculate all attention between each and every word in sentence
        # (B,  h, seq, d_k) @ (B,  h, d_k, seq) -> (B,  h, seq, seq)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Mask make sure padding or some future values should be hidden while calculating attention
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_score = torch.softmax(attention_score, dim=-1)


        if dropout is not None:
            attention_score = dropout(attention_score)

        
        # (B, h, seq, seq) @ (B, h, seq, d_k) -> (B, h, seq, d_k)
        return (attention_score @ value), attention_score


    def forward(self, q, k, v, mask):

        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v)

        #(Batch, seq, d_model) -> (Batch, seq, h, d_k) -> (B,  h, seq, d_k)
        # Transpose why? -> attention can be computed independently for each head using proper matrix multiplication
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)


        x, self.attention = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq, d_k) --> (Batch, seq, h, d_k) --> (Batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer_output):
        x = x + self.dropout(sublayer_output)
        return self.norm(x)


class EncoderBlock(nn.Module):
    def __init__(self, seq_len, d_model, h, dropout, d_ff):
        super().__init__()

        self.multiHeadAttention = MultiHeadAttention(d_model, h, dropout)
        self.norm = LayerNormalization(d_model)
        self.FeedForward = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask):
        
        att_score = self.multiHeadAttention(x, x, x, mask)
        norm_output = self.residual1(x, att_score)

        f_out = self.FeedForward(norm_output)
        norm_output = self.residual2(norm_output, f_out)

        return norm_output


class Encoder(nn.Module):
    def __init__(self, d_model, layers: nn.ModuleList):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model, h, dropout, d_ff):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, h, dropout)
        self.cross_attention = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)

        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_out, src_mask, tgt_mask):

        #  1 Masked Self Attention
        x = self.residual1(
            x,
            self.self_attention(x, x, x, tgt_mask)
        )

        # 2️ Cross Attention
        x = self.residual2(
            x,
            self.cross_attention(x, encoder_out, encoder_out, src_mask)
        )

        # 3️ Feed Forward
        x = self.residual3(
            x,
            self.feed_forward(x)
        )

        return x
    


class Decoder(nn.Module):
    def __init__(self, d_model, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        """
        x            -> (Batch, tgt_seq_len, d_model)
        encoder_out  -> (Batch, src_seq_len, d_model)
        src_mask     -> (Batch, 1, 1, src_seq_len)
        tgt_mask     -> (Batch, 1, tgt_seq_len, tgt_seq_len)
        """

        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)

        return self.norm(x)



class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, Vocab_Size)
        return torch.log_softmax(self.proj(x), dim=-1)



class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        src_pos: PositionEncoding,
        tgt_pos: PositionEncoding,
        projection_layer: ProjectionLayer
    ) :
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size, tar_vocab_size, src_seq_len, tar_seq_len, d_model = 512,
                      d_ff = 2048,
                      N = 6,
                      h = 8,
                      dropout = 0.1):
    # Create Embedding
    src_emd = InputEmbedding(d_model, src_vocab_size)
    tar_emd = InputEmbedding(d_model, tar_vocab_size)

    # Position Embedding
    src_pos = PositionEncoding(src_seq_len, d_model, dropout)
    tar_pos = PositionEncoding(tar_seq_len, d_model, dropout)

    # Create Encoder Block
    encoder_blocks = []
    for _ in range(N):
        encoder_block = EncoderBlock(
            src_seq_len,
            d_model,
            h,
            dropout,
            d_ff
        )
        encoder_blocks.append(encoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # Create Decoder Block
    decoder_blocks = []
    for _ in range(N):
        decoder_block = DecoderBlock(
            d_model=d_model,
            h=h,
            dropout=dropout,
            d_ff=d_ff
        )
        decoder_blocks.append(decoder_block)

    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Projection Layer
    projection_layer = ProjectionLayer(d_model, tar_vocab_size)

    # Transformer
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_emd,
        tgt_embed=tar_emd,
        src_pos=src_pos,
        tgt_pos=tar_pos,
        projection_layer=projection_layer
    )

    #  Initialize parameters (Xavier initialization)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

