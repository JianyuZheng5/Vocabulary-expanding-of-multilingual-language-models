#这一版仅是对【run_v2.0版】文件的路径名改动了下，其他基本没变动
import torch
import os
import numpy as np
import logging
import pickle
from tqdm.auto import tqdm
import math
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import fasttext.util
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from transformers import AutoModelForMaskedLM, AutoTokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset


#os.environ["CUDA_VISIBLE_DEVICES"]="1"   #调试的时候加上

def softmax(x, axis=-1):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


class WordEmbedding:
    """
    Uniform interface to fastText models and gensim Word2Vec models.
    """

    def __init__(self, model):
        self.model = model

        if isinstance(model, fasttext.FastText._FastText):
            self.kind = "fasttext"
        elif isinstance(model, Word2Vec):
            self.kind = "word2vec"
        else:
            raise ValueError(
                f"{model} seems to be neither a fastText nor Word2Vec model."
            )

    def has_subword_info(self):
        return self.kind == "fasttext"

    def get_words_and_freqs(self):
        if self.kind == "fasttext":
            return self.model.get_words(include_freq=True, on_unicode_error="ignore")
        elif self.kind == "word2vec":
            return (self.model.wv.index_to_key, self.model.wv.expandos["count"])

    def get_dimension(self):
        if self.kind == "fasttext":
            return self.model.get_dimension()
        elif self.kind == "word2vec":
            return self.model.wv.vector_size

    def get_word_vector(self, word):
        if self.kind == "fasttext":
            return self.model.get_word_vector(word)
        elif self.kind == "word2vec":
            return self.model.wv[word]

    def get_word_id(self, word):
        if self.kind == "fasttext":
            return self.model.get_word_id(word)
        elif self.kind == "word2vec":
            return self.model.wv.key_to_index.get(word, -1)



def load_embeddings(identifier: str, verbose=True):
    """
    Utility function to download and cache embeddings from https://fasttext.cc.

    Args:
        identifier: 2-letter language code.

    Returns:
        fastText model loaded from https://fasttext.cc/docs/en/crawl-vectors.html.
    """
    if os.path.exists(identifier):
        path = Path(identifier)
    else:
        logging.info(
            f"Identifier '{identifier}' does not seem to be a path (file does not exist). Interpreting as language code."
        )

        path = CACHE_DIR / f"cc.{identifier}.300.bin"

        #这段先注释掉
        ''' 
        if not path.exists():
            path = download(
                f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{identifier}.300.bin.gz",
                CACHE_DIR / f"cc.{identifier}.300.bin.gz",
                verbose=verbose,
            )
            path = gunzip(path)
        '''
    return fasttext.load_model(str(path))


def create_target_embeddings(
    source_subword_embeddings, target_subword_embeddings,
    source_tokens,target_tokens,
    source_matrix, neighbors=10, temperature=0.1, verbose=True):
    def get_n_closest(token_id, similarities, top_k):
        if (target_subword_embeddings[token_id] == 0).all():
            return None

        best_indices = np.argpartition(similarities, -top_k)[-top_k:]  #这句话意味着 取similarity中相似度最高的top_k个元素的下标
        best_tokens = [source_tokens[i].strip() for i in best_indices]

        best = sorted(
            [(token, similarities[idx]) for token, idx in zip(best_tokens, best_indices)],
            key=lambda x: -x[1],
        )
        return best   #返回(token, similarities[idx])的一个序列，相似度由高到低排序

    source_vocab = source_tokens

    target_matrix = np.zeros(
        (len(target_tokens), source_matrix.shape[1]), dtype=source_matrix.dtype)

    mean, std = (source_matrix.mean(0), source_matrix.std(0))

    random_fallback_matrix = np.random.RandomState(1234).normal(
        mean, std, (len(target_tokens), source_matrix.shape[1]))

    batch_size = 1024
    n_matched = 0

    not_found = []
    sources = {}

    for i in tqdm(range(int(math.ceil(len(target_matrix) / batch_size))), disable=not verbose):
        start, end = (i * batch_size, min((i + 1) * batch_size, len(target_matrix)))

        similarities = cosine_similarity(
            target_subword_embeddings[start:end], source_subword_embeddings)
        for token_id in range(start, end):
            closest = get_n_closest(token_id, similarities[token_id - start], neighbors)

            if closest is not None:
                tokens, sims = zip(*closest)
                weights = softmax(np.array(sims) / temperature, 0)
                sources[target_tokens[token_id]] = (tokens, weights, sims)
                emb = np.zeros(target_matrix.shape[1])

                for i, close_token in enumerate(tokens):
                    emb += source_matrix[source_vocab.index(close_token)] * weights[i]

                target_matrix[token_id] = emb
                n_matched += 1
            else:
                target_matrix[token_id] = random_fallback_matrix[token_id]
                not_found.append(target_tokens[token_id])

    #没啥用 注释掉了
    '''
    for token in source_tokenizer.special_tokens_map.values():
        if isinstance(token, str):
            token = [token]

        for t in token:
            if t in target_tokenizer.vocab:
                target_matrix[target_tokenizer.vocab[t]] = source_matrix[source_tokenizer.vocab[t]]
    '''
    logging.info(f"Matching token found for {n_matched} of {len(target_matrix)} tokens.")
    return target_matrix, not_found, sources




def get_subword_embeddings_in_word_embedding_space(
    tokens, model, max_n_word_vectors=None, use_subword_info=True, verbose=True):

    words, freqs = model.get_words_and_freqs()

    if max_n_word_vectors is None:
        max_n_word_vectors = len(words)

    sources = {}
    embs_matrix = np.zeros((len(tokens), model.get_dimension()))

    if use_subword_info:
        if not model.has_subword_info():
            raise ValueError("Can not use subword info of model without subword info!")

        for i in range(len(tokens)):
            token = tokens[i].strip()

            # `get_word_vector` returns zeros if not able to decompose
            embs_matrix[i] = model.get_word_vector(token)
    else:
        embs = {value: [] for value in tokenizer.get_vocab().values()}

        for i, word in tqdm(
            enumerate(words[:max_n_word_vectors]),
            total=max_n_word_vectors,
            disable=not verbose,
        ):
            for tokenized in [
                tokenizer.encode(word, add_special_tokens=False),
                tokenizer.encode(" " + word, add_special_tokens=False),
            ]:
                for token_id in set(tokenized):
                    embs[token_id].append(i)

        for i in range(len(embs_matrix)):
            if len(embs[i]) == 0:
                continue

            weight = np.array([freqs[idx] for idx in embs[i]])
            weight = weight / weight.sum()

            vectors = [model.get_word_vector(words[idx]) for idx in embs[i]]

            sources[tokenizer.convert_ids_to_tokens([i])[0]] = embs[i]
            embs_matrix[i] = (np.stack(vectors) * weight[:, np.newaxis]).sum(axis=0)

    return embs_matrix, sources







class WECHSEL:
    def _compute_align_matrix_from_dictionary(self, source_embeddings, target_embeddings, dictionary):
        correspondences = []
        idd = []

        for source_word, target_word in dictionary:
            for src_w in (source_word, source_word.lower(), source_word.title()):
                for trg_w in (target_word, target_word.lower(), target_word.title()):
                    src_id = source_embeddings.get_word_id(src_w)
                    trg_id = target_embeddings.get_word_id(trg_w)

                    if src_id != -1 and trg_id != -1 and [src_id, trg_id] not in idd:
                        idd.append([src_id, trg_id])
                        correspondences.append(
                            [
                                source_embeddings.get_word_vector(src_w),
                                target_embeddings.get_word_vector(trg_w),
                            ]
                        )

        correspondences = np.array(correspondences)
        align_matrix, _ = orthogonal_procrustes(correspondences[:, 0], correspondences[:, 1])
        return align_matrix


    def __init__(self, source_embeddings, target_embeddings,
        align_strategy="bilingual_dictionary", bilingual_dictionary=None):
        """
        Args:
            source_embeddings: fastText model or gensim Word2Vec model in the source language.

            target_embeddings: fastText model or gensim Word2Vec model in the source language.
            align_strategy: either of "bilingual_dictionary" or `None`.
                - If `None`, embeddings are treated as already aligned.
                - If "bilingual dictionary", a bilingual dictionary must be passed
                    which will be used to align the embeddings using the Orthogonal Procrustes method.
            bilingual_dictionary: path to a bilingual dictionary. The dictionary must be of the form
                ```
                english_word1 \t target_word1\n
                english_word2 \t target_word2\n
                ...
                english_wordn \t target_wordn\n
                ```
                alternatively, pass only the language name, e.g. "german", to use a bilingual dictionary
                stored as part of WECHSEL (https://github.com/CPJKU/wechsel/tree/main/dicts).
        """
        source_embeddings = WordEmbedding(source_embeddings)
        target_embeddings = WordEmbedding(target_embeddings)

        min_dim = min(
            source_embeddings.get_dimension(), target_embeddings.get_dimension()
        )
        if source_embeddings.get_dimension() != min_dim:
            fasttext.util.reduce_model(source_embeddings.model, min_dim)
        if target_embeddings.get_dimension() != min_dim:
            fasttext.util.reduce_model(source_embeddings.model, min_dim)

        if align_strategy == "bilingual_dictionary":
            if bilingual_dictionary is None:
                raise ValueError(
                    "`bilingual_dictionary` must not be `None` if `align_strategy` is 'bilingual_dictionary'."
                )
            #这段注释掉了
            '''
            if not os.path.exists(bilingual_dictionary):
                bilingual_dictionary = download(
                    f"https://raw.githubusercontent.com/CPJKU/wechsel/main/dicts/data/{bilingual_dictionary}.txt",
                    CACHE_DIR / f"{bilingual_dictionary}.txt",
                )
            '''

            dictionary = []

            for line in open(bilingual_dictionary):
                line = line.strip()
                try:
                    source_word, target_word = line.split("\t")
                except ValueError:
                    source_word, target_word = line.split()

                dictionary.append((source_word, target_word))

            align_matrix = self._compute_align_matrix_from_dictionary(
                source_embeddings, target_embeddings, dictionary
            )
            self.source_transform = lambda matrix: matrix @ align_matrix
            self.target_transform = lambda x: x
        elif align_strategy is None:
            self.source_transform = lambda x: x
            self.target_transform = lambda x: x
        else:
            raise ValueError(f"Unknown align strategy: {align_strategy}.")

        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings


    def apply(
        self, source_tokens, target_tokens, source_matrix,
        use_subword_info=True, max_n_word_vectors=None, neighbors=10, temperature=0.1):
        """
        Applies WECHSEL to initialize an embedding matrix.

        Args:
            source_tokenizer: T^s, the tokenizer in the source language.
            target_tokenizer: T^t, the tokenizer in the target language.
            source_matrix: E^s, the embeddings in the source language.
            use_subword_info: Whether to use fastText subword information. Default true.
            max_n_word_vectors: Maximum number of vectors to consider (only relevant if `use_subword_info` is False).

        Returns:
            target_matrix: The embedding matrix for the target tokenizer.
            info: Additional info about word sources, etc.
        """
        (source_subword_embeddings, source_subword_sources) \
            = get_subword_embeddings_in_word_embedding_space(
            source_tokens,
            self.source_embeddings,
            use_subword_info=use_subword_info,
            max_n_word_vectors=max_n_word_vectors)
        (target_subword_embeddings, target_subword_sources) \
            = get_subword_embeddings_in_word_embedding_space(
            target_tokens,
            self.target_embeddings,
            use_subword_info=use_subword_info,
            max_n_word_vectors=max_n_word_vectors)

        # align
        source_subword_embeddings = self.source_transform(source_subword_embeddings)
        target_subword_embeddings = self.target_transform(target_subword_embeddings)

        source_subword_embeddings /= (
            np.linalg.norm(source_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )
        target_subword_embeddings /= (
            np.linalg.norm(target_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )

        target_matrix, not_found, sources = create_target_embeddings(
            source_subword_embeddings, target_subword_embeddings,
            source_tokens, target_tokens,
            source_matrix.copy(),
            neighbors=neighbors,
            temperature=temperature,
        )

        return target_matrix

########################################################################
def read_text(text):
    with open(text, "r", encoding='utf-8') as fp:
        lines =fp.readlines()
    lines = [line.strip() for line in lines]
    print('Num of Sent:', len(lines))
    return lines

def train_val_split(filename, split_ratio):
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

def batch_iterator(batch_size=100000):
    for i in tqdm(range(0, len(target_data), batch_size)):
        yield target_data[i : i + batch_size]

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






if __name__ =='__main__':
    model_name = "./../bert-base-multilingual-cased"
    target_data_filename = "am.txt"                          #改一下
    saved_path = './' + target_data_filename[:2] + '_mBERT'
    train_file_path = target_data_filename[:2] + '_train.txt'
    valid_file_path = target_data_filename[:2] + '_val.txt'
    vocab_size = 30000
    split_ratio = 0.9

    # 注：用AutoModelForMaskedLM 有些关于NSP的权重没有导入进来
    # solution:https://github.com/huggingface/transformers/issues/6646 (可能不太管用)
    source_tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    target_data = read_text(target_data_filename)
    train_val_split(target_data_filename, split_ratio)

    iter_dataset = iter(target_data)
    # train_new_from_iterator
    # https://github.com/huggingface/notebooks/blob/main/examples/tokenizer_training.ipynb
    target_tokenizer = source_tokenizer.train_new_from_iterator(
        batch_iterator(),
        vocab_size=vocab_size
    )

    new_vocab = []
    source_vocab = list(source_tokenizer.vocab.keys())
    target_vocab = list(target_tokenizer.vocab.keys())
    for subword in target_vocab:
        if subword not in source_vocab:
            new_vocab.append(subword)

    # 教程:https://zhuanlan.zhihu.com/p/391814780
    num_added_toks = source_tokenizer.add_tokens(new_vocab)
    model.resize_token_embeddings(len(source_tokenizer))
    source_tokenizer.save_pretrained(saved_path)

    print("新加词：", len(new_vocab))
    print("Vocabulary size:", len(source_tokenizer))
    print("Embedding layer size:", model.get_input_embeddings().weight.detach().cpu().numpy().shape)



    source_language_emb = './../fasttext_embeddings/cc.en.300.bin'
    target_language_emb = './../fasttext_embeddings/cc.am.300.bin'            #改一下
    bilingual_dictionary = './am_bilingual_dic1.txt'      #改一下


    wechsel = WECHSEL(
        load_embeddings(source_language_emb),
        load_embeddings(target_language_emb),
        bilingual_dictionary=bilingual_dictionary
    )
    print("wechsel完毕")


    source_tokens = []
    target_tokens = new_vocab
    source_matrix = []
    with open('source_tokens.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    source_tokens = [line.strip() for line in lines]

    with open('source_matrix.dat', 'rb') as f:
        source_matrix = pickle.load(f)


    target_matrix = wechsel.apply(
        source_tokens,
        target_tokens,
        source_matrix
   )


    '''
    with open('target_matrix(sim).dat', 'wb') as fp:
        pickle.dump(target_matrix, fp)
    with open('target_matrix(ini).dat', 'wb') as fp:
        pickle.dump(model.get_input_embeddings().weight.detach().cpu().numpy()[119547:], fp)
    '''



    train_dataset = LineByLineTextDataset(
        tokenizer=source_tokenizer,
        file_path=train_file_path,  # 注 mention train text file here
        block_size=128)  # EMNLP2020那个工作说采用训练BERT的默认设置，是128

    valid_dataset = LineByLineTextDataset(
        tokenizer=source_tokenizer,
        file_path=valid_file_path,  # 注 mention valid text file here
        block_size=128)  # EMNLP2020那个工作说采用训练BERT的默认设置，是128

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=source_tokenizer, mlm=True, mlm_probability=0.15)

    # 参数详解:https://blog.csdn.net/duzm200542901104/article/details/132762582
    training_args = TrainingArguments(
        output_dir="./" + target_data_filename[:2] + "_checkpoint",  # select model path for checkpoint
        overwrite_output_dir=False,
        num_train_epochs=10,            #改一下
        per_device_train_batch_size=32,  ##默认是8,EMNLP2020那篇是32
        per_device_eval_batch_size=32,  ##默认是8，EMNLP2020那篇是32
        gradient_accumulation_steps=1,  ##默认是1
        evaluation_strategy='no',  ##默认是no，可选 epoch、step
        save_strategy="no",  ##保存checkpoint
        save_total_limit=3,  ##限制保存的checkpoint的总数量
        eval_steps=1000,  ##
        load_best_model_at_end=False,  ##
        metric_for_best_model='eval_loss',  ##
        greater_is_better=False,  ##
        prediction_loss_only=False,  ##当执行评估和预测的时候，是否仅返回损失
        report_to="none",  ##报告结果和日志的integration列表，默认是all，我改成了none
        learning_rate=2e-5,  ##默认是5e-5，EMNLP2020那篇是2e-5
        weight_decay=0,  ##默认是0
        lr_scheduler_type='linear',  ##选择什么类型的学习率调度器来更新模型的学习率，默认是linear
        optim='adamw_hf',  ##默认使用adamw_hf
        group_by_length='False',  ##是否将训练数据集中长度大致相同的样本分组在一起,默认是False
        length_column_name='length',  ##预计算列名的长度(好像没啥用)
        seed=2024,
        # logging_dir = './logs',
        # logging_strategy = 'steps'       #训练期间采用的日志策略,默认为steps
        # label_names = ['label']
        # label_smoothing_factor=0
    )

    # 早停：https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )

    #save_model与save_pretrained的区别https://stackoverflow.com/questions/72108945/saving-finetuned-model-locally
    trainer.train()
    model.save_pretrained(saved_path)



