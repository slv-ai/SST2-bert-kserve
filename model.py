import os
import numpy as np 
import pandas as pd 
from datasets import load_dataset
import torch 
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score
import argparse
import json

torch.manual_seed(42)
np.random.seed(42)

def load_and_prepare_data():
    print("loading the SST-2 dataset...")
    dataset=load_dataset("glue","sst2")
    train_data=dataset["train"]
    eval_data=dataset["validation"]
    return train_data,eval_data

def tokenize_data(tokenizer,texts,labels,max_length=128):
    encodings=tokenizer(texts,truncation=True,padding="max_length",
                        max_length=max_length,return_tensors='pt')

    class Sentiment(torch.utils.data.Dataset):
        def __init__(self,encodings,labels):
            self.encodings=encodings
            self.labels=labels
        def __getitem__(self,idx):
            item={key:val[idx] for key,val in self.encodings.items()}
            item['labels']=torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)
    dataset=Sentiment(encodings,labels)
    return dataset

def train_model(model,train_dataloader,eval_dataloader,device,
                epochs=3,save_dir="./model_output"):
    optimizer=AdamW(model.parameters(),lr=2e-5)           
    total_steps=len(train_dataloader)*epochs # total no of training across all epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)

    best_accuracy=0
    for epoch in epochs:
        print(f"\n Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss=0

        for batch in train_dataloader:
            model.zero_grad()
            input_ids=batch['input_ids'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            labels=batch['labels'].to(device)

            outputs=model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
            loss=outputs.loss
            total_loss += loss.item()

            loss.backward() #compute gradients
            torch.nn.utils.clip_grad_norm(model.parameters(),1.0) #prevent exploding gradients
            optimizer.step() #update gradients
            scheduler.step() #update learning rate
        avg_training_loss=total_loss / len(train_dataloader)
        print(f"avg training loss {avg_training_loss:.4f}")

        accuracy=evaluate_model(model,eval_dataloader,device)
        print(f"validation accuracy :{accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print(f"saving best model with accuracy :{accuracy:.4f}")
            model.save_pretrained(save_dir)

            #save model config
            model_config={
                "model_name" :"bert-base-uncased",
                "task": "sentiment-analysis",
                "classes" : ["negative","positive"],
                "accuracy" : float(accuracy)
            }
            with open(os.path.join(save_dir,"model_config.json"),'w') as f_in:
                json.dump(model_config,f_in)
        return best_accuracy

def evaluate_model(model,Dataloader,device):
    model.eval()
    predictions=[]
    actual_labels=[]
    with torch.no_grad():
        for batch in dataloader:
            input_ids=batch['input_ids'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            labels=batch['labels'].to(device)

            outputs=model(input_ids=input_ids,attention_mask=attention_mask)
            logits=outputs.logits

            preds=torch.argmax(logits,dim=1).cpu().numpy()
            actuals=labels.cpu().numpy()
            predictions.extend(preds)
            actual_labels.extend(actuals)
    accuracy=accuracy_score(actual_labels,predictions)
    return accuracy



def main():
    parser=argparse.ArgumentParser(description="fine-tune BERT for sentiment analysis")
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='no. of epochs to train')
    parser.add_argument('--model_dir',type=str, default='./model_output',help='directory to save model')
    args = parser.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device using: {device}")

    #load data
    train_data,eval_data = load_and_prepare_data()
    #initialize tokenizer and model
    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    model=BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)
    model.to(device)

    #save tokenizer
    tokenizer.save_pretrained(args.model_dir)

    #tokenize data
    train_dataset=tokenize_data(tokenizer,train_data['sentence'],train_data['label'])
    eval_dataset=tokenize_data(tokenizer,eval_data['sentence'],eval_data['label'])

    #create dataloaders
    train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    eval_dataloader=DataLoader(eval_dataset,batch_size=args.batch_size)

    #train the model
    accuracy=train_model(model,train_dataloader,eval_dataloader,device,epochs=args.epochs,save_dir=args.model_dir)
    

    print(f"training complete with best accuracy:{accuracy : .4f}")
    print(f"model saved to {args.model_dir}")

if __name__ == 'main':
    main()

        

