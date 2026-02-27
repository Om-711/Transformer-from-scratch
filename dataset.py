import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Casual Maks make sure it will not see future words
def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

class BilingualDataset(Dataset):
    def __init__(self, ds_raw, src_tokenizer, tar_tokenizer, src_lang, tar_lang, seq_len):
        super().__init__()

        self.ds_raw = ds_raw
        self.src_tokenizer = src_tokenizer
        self.tar_tokenizer = tar_tokenizer
        self.src_lang = src_lang
        self.tar_lang = tar_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([self.src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([self.src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([self.src_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)


    def __len__(self):
        return len(self.ds_raw)
    
    def __getitem__(self, index):
        # Extract all the pairs from the hugging face then take source and tar text from it
        src_target_pairs = self.ds_raw[index]
        src_text = src_target_pairs['translation'][self.src_lang]
        tar_text = src_target_pairs['translation'][self.tar_lang]
        
        # Converts text -> into list of word IDs (numbers)
        enc_input_token = self.src_tokenizer.encode(src_text).ids 
        dec_input_token = self.tar_tokenizer.encode(tar_text).ids 

        enc_num_padding_token = self.seq_len - len(enc_input_token) - 2 # 2 for SOS and EOS
        dec_num_padding_token = self.seq_len - len(dec_input_token) - 1 # for SOS

        # .Add SOS EOS and padding to input
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_token, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_token, dtype=torch.int64)
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_token, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_token, dtype=torch.int64)
        ])

        label = torch.cat([
            torch.tensor(dec_input_token, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_token, dtype=torch.int64)
        ])


        return {
            "encoder_input" : encoder_input, # (SeqLen)
            "decoder_input" : decoder_input, # (SeqLen)
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #[1, 1, SeqLen]
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & (casual_mask(decoder_input.size(0))), #[1, 1, SeqLen] [1,SeqLen, SeqLen]
            "label":label,
            "src_text" : src_text,
            "tar_text" : tar_text
        }


