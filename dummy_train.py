from datasets import load_dataset
import torch
imdb = load_dataset("imdb")
print(imdb["train"][0])

from transformers import (
    AutoTokenizer, DataCollatorWithPadding, 
    AutoModelForSequenceClassification, 
    TrainingArguments, Trainer,
    DataCollatorForTokenClassification
)

# model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
model_name = 'distilbert-base-uncased'

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized_imdb = imdb.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=60,
    per_device_eval_batch_size=60,
    num_train_epochs=10000000,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()