import torch
import pandas as pd
import random
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from dataset import Data
from models.model import Model

# Hyperparameters
hyp ={
    "lr" : 5e-5,
    "epochs" : 15,
    "batch_size" : 64,
    "seed" : 123,
    "step" : 100,
    "max_input_length" : 512,
    "max_target_length" :128
}
    
def train():
    model = Model(lr=hyp["lr"])

    # Optimizer and tokenizer 
    tokenizer = model.tokenizer
    optimizer = model.configure_optimizers()


    # Readfile and make to dataloader
    filepath = "../data/processed/"
    df_train = pd.read_csv(filepath+'train.csv')
    df_valid = pd.read_csv(filepath+'valid.csv')    
    train = Data(df_train ,tokenizer,hyp["max_target_length"])
    valid = Data(df_valid ,tokenizer,hyp["max_target_length"])
    train_dataloader = DataLoader(train, batch_size =hyp["batch_size"])
    valid_dataloader = DataLoader(valid, batch_size =hyp["batch_size"])

    # Losses list
    train_losses = []
    valid_losses = []

    
    for e in range(hyp["epochs"]):
        train_loss = 0
        running_loss = 0
        model.train()
        print("Epoch: {}/{}.. ".format(e + 1, hyp["epochs"]))
        for steps, batch in enumerate(train_dataloader):
            # load data and labels in the batch
            input_ids = batch[0]
            masks = batch[1]
            labels = batch[2]

            # Training
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=masks, labels=labels)
            # Output and loss
            loss = outputs.loss
            logits = outputs.logits
            running_loss += loss.item()
            train_loss += loss.item()

            # Print loss after some steps
            if steps % hyp["step"] == 0 and not steps == 0:
                # Print generated value
                # original_text = tokenizer.decode(labels[0], skip_special_tokens=True)
                # print("Original Title:", original_text)
                # outputs = model.generate(input_ids)
                # print("Generate title:",outputs)
                print(
                    "Batch: {}/{}.. ".format(steps, len(train_dataloader)),
                    "Training Loss: {:.3f}.. ".format(running_loss / hyp["step"]))
                running_loss = 0
                
            loss.backward()
            optimizer.step()

        # Validating    
        valid_loss = valid(model, valid_dataloader)
        print(
            "Training Loss: {:.3f}.. ".format(train_loss / len(train_dataloader)),
            "Valid Loss: {:.3f} ".format(valid_loss),)
        valid_losses.append(valid_loss)
        train_losses.append(train_loss / len(train_dataloader))
#     torch.save(model.state_dict(), 'models/trained_model.pt')


def valid(model, valid_dataloader):
    model.eval()
    
    running_loss = 0
    for batch in valid_dataloader :
        input_ids = batch[0]
        masks = batch[1]
        labels = batch[2]
        # Turn off gradients for validation, will speed up inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=masks, labels=labels)
            loss = outputs.loss
        running_loss += loss.item()

    random_batch = random.choice(list(valid_dataloader))

    
    original_text = tokenizer.decode(random_batch[2][0], skip_special_tokens=True) 
    print("Original Title:", original_text)
    outputs = model.generate(random_batch[0])
    print("Generate title:",outputs)

    return(running_loss/len(valid_dataloader))
    
    
    

if __name__ == "__main__":
    train()