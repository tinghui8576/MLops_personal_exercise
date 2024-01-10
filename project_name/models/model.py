import torch

from transformers import T5ForConditionalGeneration, AdamW, T5Tokenizer


class Model(torch.nn.Module):
    def __init__(self, lr=5e-5):
        super().__init__()
        
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):     
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def generate(self, input_ids):
        # Use the generate method from the T5 model
        generated_ids = self.model.generate(input_ids)
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
    
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        
        return AdamW(self.parameters(), lr=self.lr)
