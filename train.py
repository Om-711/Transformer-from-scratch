import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

from dataset import BilingualDataset, casual_mask
from model import build_transformer

from config import get_config, get_weights_file_path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_build_tokenizer(config, ds, lang):

    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        # Creata Word level tokenizer and keep unkown words as UNK
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace() #This splits text by spaces.
        # This trains vocabulary from dataset. [UNK] -> unknown word, [PAD] -> padding
        # [SOS] -> start of sentence, [EOS] -> end of sentence
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )
        # It reads all sentences from dataset ds in language lang and builds vocabulary.
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang),
            trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tar']}", split='train')

    # Build tokenizer 
    tokenizer_src = get_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tar = get_build_tokenizer(config, ds_raw, config['lang_tar'])

    # Keep 90% train and 10% test
    train_ds_size = int(0.9 * len(ds_raw))
    test_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, test_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tar, config['lang_src'], config['lang_tar'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tar, config['lang_src'], config['lang_tar'], config['seq_len'])

    # Now we want to know the maximum length of the dataset to define for max length
    max_src_len = 0
    max_tar_len = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tar_ids = tokenizer_tar.encode(item['translation'][config['lang_tar']]).ids

        max_src_len = max(max_src_len, len(src_ids))
        max_tar_len = max(max_tar_len, len(tar_ids))

    print(f"Maximum length of Source Sentence:{max_src_len}")
    print(f"Maximum length of Target Sentence:{max_tar_len}")


    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    return train_loader, test_loader, tokenizer_src, tokenizer_tar


def get_model(config, src_vocab_size, tar_vocab_size):
    model = build_transformer(src_vocab_size, tar_vocab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model



def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])

        # Select the token with the max probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)

        # decoder input + next word will be input for next one
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
            ],
            dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)
    
def run_validation(config, model, validation_ds, tokenizer_src, tokenizer_tar, max_len, device, print_msg, global_state, writer, num_example=2):
    model.eval()
    count += 1

    source_text = []
    expected_text = []
    predicted = []

    # Size of the control window(default rk)
    console_width = 50

    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch['encoder_input'].to(device)  # (B, Seq_Len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, Seq_Len)

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tar, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tar.decode(model_out.detach().cpu().numpy())

            source_text.append(source_text)
            expected_text.append(target_text)
            predicted.append(model_out_text)

            # Print to the console
            print_msg('-' * console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_example:
                break




def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    config = get_config()
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)


    train_loader, test_loader, tokenizer_src, tokenizer_tar = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tar.get_vocab_size())

    # Create Summary of it
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)


    initial_epoch = 0
    global_step = 0

    # Label smoothing take 0.1 percent of score from highest probaility and share it equally to all other 
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        
        batch_iterator = tqdm(train_loader, desc=f'Processing epoch {epoch:02d}')
        
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device)  # (B, Seq_Len)
            decoder_input = batch['decoder_input'].to(device)  # (B, Seq_Len)
            encoder_mask = batch['encoder_mask'].to(device)    # (B, 1, 1, Seq_Len)
            decoder_mask = batch['decoder_mask'].to(device)    # (B, 1, Seq_Len, Seq_Len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, Seq_Len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, Seq_Len, d_model)
            proj_output = model.project(decoder_output)  # (B, Seq_Len, tgt_vocab_size)

            label = batch['label'].to(device)  # (B, Seq_Len)

            # (B, Seq_Len, tgt_vocab_size) -> (B * Seq_Len, tgt_vocab_size)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tar.get_vocab_size()),
                label.view(-1)
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"}) #It updates the progress bar to display the current loss value.

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        # Checking how the model performs
        run_validation(config, model, test_loader, tokenizer_src, tokenizer_tar, config['seg_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, num_example=2)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)



if __name__ == '__main__':
    config = get_config()
    train_model()