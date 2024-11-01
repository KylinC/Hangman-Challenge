import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizerFast, DataCollatorForLanguageModeling
import string

from tqdm import tqdm
import wandb

from transformers import DataCollatorForLanguageModeling
import torch
import random
from typing import Optional

class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, max_mlm_probability=0.25):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.max_mlm_probability = max_mlm_probability

    def torch_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None):
        """
        Prepare masked tokens inputs/labels for masked language modeling
        """
        labels = inputs.clone()
        batch_size, seq_length = inputs.shape


        if special_tokens_mask is None:
            special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
            for special_token_id in self.tokenizer.all_special_ids:
                special_tokens_mask = special_tokens_mask | (inputs == special_token_id)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        for i in range(batch_size):
            input_ids = inputs[i]
            probability_matrix = torch.zeros(seq_length)
            num_tokens = (~special_tokens_mask[i]).sum().item()
            mlm_probability = random.uniform(0.0, self.max_mlm_probability)
            num_to_mask = max(1, int(mlm_probability * num_tokens))
            candidate_indices = torch.nonzero(~special_tokens_mask[i], as_tuple=False).view(-1)
            if len(candidate_indices) >= num_to_mask:
                mask_indices = random.sample(candidate_indices.tolist(), num_to_mask)
                probability_matrix[mask_indices] = 1.0
            else:
                probability_matrix[candidate_indices] = 1.0
            masked_indices = probability_matrix.bool()
            labels[i][~masked_indices] = -100 
            indices_replaced = (torch.bernoulli(torch.full(labels[i].shape, 0.8)).bool()) & masked_indices
            input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            indices_random = (torch.bernoulli(torch.full(labels[i].shape, 0.5)).bool()) & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels[i].shape, dtype=torch.long)
            input_ids[indices_random] = random_words[indices_random]

            inputs[i] = input_ids

        return inputs, labels

# two stage training
stage = 2

letters = list(string.ascii_lowercase)
special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
vocab = special_tokens + letters

if stage==1:
    tokenizer = BertTokenizerFast.from_pretrained('/archive/share/cql/model/bert-base-uncased')
    tokenizer.add_tokens(letters)
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'unk_token': '[UNK]',
        'cls_token': '[CLS]',
        'sep_token': '[SEP]',
        'mask_token': '[MASK]'
    })

    model = BertForMaskedLM.from_pretrained('/archive/share/cql/model/bert-base-uncased')
    model.resize_token_embeddings(len(tokenizer))
    wandb.init(project='hangman_bert', config={
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 5e-5,
        'mlm_probability': 0.25,
    })
else:
    tokenizer = BertTokenizerFast.from_pretrained('/archive/share/cql/trex/bert_hangman_tokenizer_random_epoch_28')
    model = BertForMaskedLM.from_pretrained('/archive/share/cql/trex/bert_hangman_model_random_epoch_28')
    wandb.init(project='hangman_bert', config={
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 2e-5,
        'mlm_probability': 0.5,
    })

class HangmanDataset(Dataset):
    def __init__(self, words):
        self.words = words

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        tokens = list(word)
        return tokens

with open('words_250000_train.txt', 'r') as f:
    words = f.read().splitlines()

train_size = int(0.95 * len(words))
train_words = words[:train_size]
val_words = words[train_size:]

train_dataset = HangmanDataset(train_words)
val_dataset = HangmanDataset(val_words)

data_collator = CustomDataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    max_mlm_probability=wandb.config.mlm_probability
)

def collate_fn(examples):
    encodings = tokenizer(examples, is_split_into_words=True, padding=True, truncation=True)
    batch = []
    for i in range(len(examples)):
        sample = {key: encodings[key][i] for key in encodings}
        batch.append(sample)
    batch = data_collator(batch)
    return batch

train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)

wandb.watch(model, log='all')

epochs = wandb.config.epochs
best_val_loss = float('inf')
best_models = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")
    for batch in train_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation")
        for batch in val_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            val_bar.set_postfix(loss=loss.item())
    avg_val_loss = val_loss / len(val_loader)

    wandb.log({
        'epoch': epoch + 1,
        'train_loss': avg_loss,
        'val_loss': avg_val_loss,
    })

    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    if stage==1:
        model_name = f'bert_hangman_model_random_epoch_{epoch+1}'
        tokenizer_name = f'bert_hangman_tokenizer_random_epoch_{epoch+1}'
    else:
        model_name = f'bert_hangman_model_random_epoch_{epoch+1}_stage2'
        tokenizer_name = f'bert_hangman_tokenizer_random_epoch_{epoch+1}_stage2'

    if len(best_models) < 3:
        model.save_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_name)
        best_models.append((avg_val_loss, epoch+1, model_name, tokenizer_name))
        best_models.sort(key=lambda x: x[0])
    else:
        if avg_val_loss < best_models[-1][0]:
            worst_model = best_models.pop(-1)
            if os.path.exists(worst_model[2]):
                os.system(f'rm -rf {worst_model[2]}')
            if os.path.exists(worst_model[3]):
                os.system(f'rm -rf {worst_model[3]}')
            model.save_pretrained(model_name)
            tokenizer.save_pretrained(tokenizer_name)
            best_models.append((avg_val_loss, epoch+1, model_name, tokenizer_name))
            best_models.sort(key=lambda x: x[0])


wandb.finish()