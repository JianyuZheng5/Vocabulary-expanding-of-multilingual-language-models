#扩充词表，并对子词做随机初始化表示后，继续预训练mBERT
#教程1：https://blog.csdn.net/pythonwyq/article/details/125464819
#教程2：https://zhuanlan.zhihu.com/p/358175926
import torch
import math
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
#from transformers import EarlyStoppingCallback, IntervalStrategy
#from transformers import BertModel, BertForMaskedLM, BertForSequenceClassification, BertTokenizer


model_name = "./../bert-base-multilingual-cased"
target_data_filename = "am.txt"
saved_path = './'+target_data_filename[:2]+'_mBERT'
train_file_path = target_data_filename[:2]+'_train.txt'
valid_file_path = target_data_filename[:2]+'_val.txt'
vocab_size = 30000
split_ratio = 0.9


def train_val_split(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    split = int(len(lines)*split_ratio)
    lines_train, lines_val = lines[:split], lines[split:]
    with open(filename[:2]+'_train.txt', 'w', encoding='utf-8') as f1:
        for line in lines_train:
            f1.write(line)
    with open(filename[:2]+'_val.txt', 'w', encoding='utf-8') as f2:
        for line in lines_val:
            f2.write(line)


def read_text(text):
    with open(text, "r", encoding='utf-8') as fp:
        lines =fp.readlines()
    lines = [line.strip() for line in lines]
    print('Num of Sent:', len(lines))
    return lines


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


#注：用AutoModelForMaskedLM 有些关于NSP的权重没有导入进来
#solution:https://github.com/huggingface/transformers/issues/6646 (可能不太管用)
source_tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)



target_data = read_text(target_data_filename)
train_val_split(target_data_filename)


iter_dataset = iter(target_data)
def batch_iterator(batch_size=100000):
    for i in tqdm(range(0, len(target_data), batch_size)):
        yield target_data[i : i + batch_size]

#train_new_from_iterator
#https://github.com/huggingface/notebooks/blob/main/examples/tokenizer_training.ipynb
target_tokenizer = source_tokenizer.train_new_from_iterator(
    batch_iterator(),
    vocab_size=vocab_size
)

#vocab = list(target_tokenizer.vocab.keys())
#with open('target_vocab.txt', 'w', encoding='utf-8') as fp:
#    for token in sorted(vocab):
#        fp.write(token+'\n')


new_vocab = []
source_vocab = list(source_tokenizer.vocab.keys())
target_vocab = list(target_tokenizer.vocab.keys())
for subword in target_vocab:
    if subword not in source_vocab:
        new_vocab.append(subword)



#教程:https://zhuanlan.zhihu.com/p/391814780
num_added_toks = source_tokenizer.add_tokens(new_vocab)
#num_added_toks = source_tokenizer.add_tokens(['fdahfj'])
model.resize_token_embeddings(len(source_tokenizer))
source_tokenizer.save_pretrained(saved_path)

print("Vocabulary size:",len(source_tokenizer))
print("Embedding layer size:",model.get_input_embeddings().weight.detach().cpu().numpy().shape)
#aa = model.get_input_embeddings().weight.detach().cpu().numpy()[-25000:]
#print(aa)



train_dataset = LineByLineTextDataset(
    tokenizer=source_tokenizer,
    file_path=train_file_path,  # 注 mention train text file here
    block_size=128)             #EMNLP2020那个工作说采用训练BERT的默认设置，是128
 
valid_dataset = LineByLineTextDataset(
    tokenizer=source_tokenizer,
    file_path=valid_file_path,   # 注 mention valid text file here
    block_size=128)              #EMNLP2020那个工作说采用训练BERT的默认设置，是128
 
data_collator = DataCollatorForLanguageModeling(
    tokenizer=source_tokenizer, mlm=True, mlm_probability=0.15)


#参数详解:https://blog.csdn.net/duzm200542901104/article/details/132762582
training_args = TrainingArguments(
    output_dir="./"+target_data_filename[:2]+"_checkpoint",  # select model path for checkpoint
    overwrite_output_dir=False,
    num_train_epochs=10,
    per_device_train_batch_size=32,     ##默认是8,EMNLP2020那篇是32
    per_device_eval_batch_size=32,      ##默认是8，EMNLP2020那篇是32
    gradient_accumulation_steps=1,      ##默认是1
    evaluation_strategy='no',           ##默认是no，可选 epoch、step
    save_strategy="no",                 ##保存checkpoint
    save_total_limit=3,                 ##限制保存的checkpoint的总数量
    eval_steps=1000,                    ##
    load_best_model_at_end=False,       ##
    metric_for_best_model='eval_loss',  ##
    greater_is_better=False,            ##
    prediction_loss_only=False,         ##当执行评估和预测的时候，是否仅返回损失
    report_to="none",                   ##报告结果和日志的integration列表，默认是all，我改成了none
    learning_rate = 2e-5,               ##默认是5e-5，EMNLP2020那篇是2e-5
    weight_decay = 0,                   ##默认是0
    lr_scheduler_type = 'linear',       ##选择什么类型的学习率调度器来更新模型的学习率，默认是linear
    optim ='adamw_hf',                  ##默认使用adamw_hf
    group_by_length='False',            ##是否将训练数据集中长度大致相同的样本分组在一起,默认是False
    length_column_name = 'length',      ##预计算列名的长度(好像没啥用)
    seed = 2024,
    #logging_dir = './logs',
    #logging_strategy = 'steps'       #训练期间采用的日志策略,默认为steps
    #label_names = ['label']
    #label_smoothing_factor=0
    )

#EMNLP2020那篇训练了50万轮


#早停：https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
)

'''
#load checkpoint：https://github.com/huggingface/trl/issues/674
from transformers.trainer_utils import EvalPrediction, get_last_checkpoint

last_checkpoint = get_last_checkpoint(script_args.output_dir)
trainer.train(resume_from_checkpoint=last_checkpoint)
'''


#save_model与save_pretrained的区别https://stackoverflow.com/questions/72108945/saving-finetuned-model-locally
trainer.train()
model.save_pretrained(saved_path)
#eval_results = trainer.evaluate()
#print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")



