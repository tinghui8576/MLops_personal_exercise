import torch
from torch.utils.data import Dataset

class wikiData(Dataset):
    def __init__(self, df, tokenizer, max_length=128):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        inputs = ["summarize:" + text for text in df["body_text"]]
        input_tokenize = tokenizer( 
                                inputs,
                                add_special_tokens=True,        #Add Special tokens like [CLS] and [SEP]
                                max_length=max_length,
                                padding = 'max_length',         #for padding to max_length for equal sequence length
                                truncation = True,              #truncate the text if it is greater than max_length
                                return_attention_mask=True,     #will return attention mask
                                return_tensors="pt"             #return tensor formate
                                )

        self.input_ids = torch.tensor(input_tokenize['input_ids'])
        self.attention_mask = torch.tensor(input_tokenize['attention_mask'])
        
        with tokenizer.as_target_tokenizer():
            label_tokenize = tokenizer(
                                    list(df["title"]), 
                                    add_special_tokens=True,        #Add Special tokens like [CLS] and [SEP]
                                    max_length=max_length,
                                    padding = 'max_length',         #for padding to max_length for equal sequence length
                                    truncation = True,              #truncate the text if it is greater than max_length
                                    return_attention_mask=True,     #will return attention mask
                                    return_tensors="pt"
                                    )
                
            self.labels = torch.tensor(label_tokenize['input_ids'])
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx] 
    