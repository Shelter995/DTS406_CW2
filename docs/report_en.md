# DTS406 Coursework Report draft in English

## Abstract

This report accomplishes two NLP tasks: the first part is response selection in dialogue systems, and the second part is document summarization. Response selection is a retrieval and ranking task, with the goal of choosing the most appropriate reply from the candidate responses when given a multi-turn dialogue context. Document summarization is a generative task, with the goal of compressing long documents into concise summaries. The two parts respectively use independent datasets, models and evaluation metrics to demonstrate the complete process from data preprocessing and model implementation to result analysis.

---

# Part One：Response Selection in Dialogue Systems

## 1. Literature Review

Dialogue systems are commonly found in customer service chatbots, technical support assistants and question-answering systems. Unlike generative dialogue, a response selection system ranks predefined candidates, thus placing more emphasis on relevance and stability. For instance, e-commerce customer service can select responses from standard after-sales scripts, and Ubuntu technical support assistants can search for existing solutions based on user error descriptions.

In practical systems, response selection is often used as the last stage of a retrieval pipeline. A first-stage retriever may collect a small set of possible responses from a large response bank, and a neural ranker then reorders them according to the dialogue context. This design is safer than pure generation in domains where factual correctness or policy consistency is important, because every final response comes from an existing candidate set. It also makes evaluation clearer: the system can be measured by whether the correct response is ranked near the top.

There are three main challenges in response selection: First, not all historical utterances are equally important in multi-turn contexts, and the model needs to identify the turns on which the current problem depends. Second, hard negative samples may share a large number of technical terms with correct responses, which can easily lead to misjudgment. Third, large-scale candidate ranking needs to take into account both efficiency and matching accuracy. Bi-encoders can pre-compute candidate vectors, which is fast but has weak interaction. Cross-encoders input `context` and `response` into the model together, which can capture token-level relationships, but the inference cost is higher.

Base LLM methods can be fine-tuned for tasks using BERT/MiniLM-style encoders. The instruction-tuned LLM method enables the model to score candidate responses through prompts. The former has a clear structure and is suitable for supervised training, while the latter has strong comprehension ability but is slow and may produce unstable output formats. In this experiment, the base MiniLM cross-encoder was selected as the main method.

## 2. Data and Preprocessing

This experiment uses Ubuntu Dialogue Corpus v2.0. The training set is in the binary classification format `Context, Utterance, Label`, and the validation and test sets are in the 1-of-10 ranking format, that is, 1 ground-truth response and 9 distractor responses. The experimental samples were `train_100k`, `valid_5k` and `test_5k`. The candidate order for validation and testing is shuffled using a fixed seed, and the correct response index is saved.

The model input retains `__eou__` and `__eot__` to preserve utterance and turn boundaries. Stop words are not deleted from the training input because negation words and function words are important for semantic matching. In the statistical analysis stage, lowercasing, word tokenization and stopword removal are used.

The training and evaluation formats are intentionally different. The training set is pairwise, so the model learns a binary matching function over individual context-response pairs. The validation and test sets are listwise ranking problems, so the same matching function is applied to each candidate and the candidates are sorted by score. This mirrors a real retrieval-based dialogue system: a classifier can be trained on positive and negative pairs, but its final usefulness is judged by ranking quality. The 1:1 positive-negative balance in training also prevents the classifier from becoming biased toward the negative class.

| Split | Sample size |  Average context tokens | Average response/candidate tokens | Vocabulary list size |
|---|------------:|------------------------:|----------------------------------:|---:|
| train_100k |      100000 |                   75.07 |                             14.94 | 114692 |
| valid_5k |        5000 |                   80.24 |                             17.23 | 27515 |
| test_5k |        5000 |                   82.96 |                             17.21 | 27886 |

## 3. Model and Implementation

The model is `microsoft/MiniLM-L12-H384-uncased`. The input is encoded by the tokenizer as `[CLS] context [SEP] response [SEP]`, with a binary classification head added on top, and the output is match/not-match logits. During testing, the positive-class logit is calculated for the 10 candidates of the same context, and candidates are sorted by score.

MiniLM was chosen because it is a compact Transformer encoder derived from BERT-style models through distillation. Compared with BERT-base, it has a smaller hidden size and lower computation cost, while still preserving strong language understanding ability. This makes it appropriate for a coursework-scale experiment with 100,000 training pairs. A cross-encoder was preferred over a bi-encoder because the task uses only ten candidates per context during evaluation, so the extra inference cost is acceptable and the stronger context-response interaction is useful.

```text
for (context, response, label) in train:
    x = tokenizer(context, response, max_length=256)
    logits = MiniLM(x)
    loss = CE(logits, label)
    update parameters

for each test context:
    score_i = positive_logit(MiniLM(context, candidate_i))
    rank candidates by score_i
    compute MRR and Recall@k
```

The training parameters are set to epochs 3, batch size 32, learning rate 2e-5, warmup ratio 0.1, weight decay 0.01, and BF16 is preferred. Each round is evaluated on valid_5k, and the best checkpoint is selected by MRR.

MRR and Recall@k were selected because response selection is a ranking task. Recall@1 measures whether the top-ranked response is correct, while Recall@5 measures whether the model can place the correct response in a useful shortlist. MRR adds more detail by rewarding higher correct ranks, so a correct response at rank 2 receives more credit than one at rank 8.

## 4. Result Analysis

`figure 1：outputs/part_one/results/figures/validation_metrics.png`  
`figure 2：outputs/part_one/results/figures/correct_rank_distribution.png`  
`figure 3：outputs/part_one/results/figures/recall_at_k.png`  
`figure 4：outputs/part_one/results/figures/score_gap_distribution.png`

| Epoch | Loss | MRR | Recall@1 | Recall@2 | Recall@5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5412 | 0.7220 | 0.5982 | 0.7218 | 0.8924 |
| 2 | 0.4474 | 0.7347 | 0.6172 | 0.7340 | 0.8958 |
| 3 | 0.3957 | 0.7412 | 0.6224 | 0.7476 | 0.9036 |

| Test Metric | Score |
|---|---:|
| MRR | 0.7492 |
| Recall@1 | 0.6336 |
| Recall@2 | 0.7508 |
| Recall@5 | 0.9124 |

The results show that the model can rank the candidate responses well on the test set. Recall@1 is 0.6336, indicating that correct responses rank first in approximately 63% of the samples; Recall@5 reaches 0.9124, indicating that most correct responses can enter the top five. The MRR is 0.7492, indicating a high overall ranking. The validation curve also shows a consistent improvement from epoch 1 to epoch 3, so the final checkpoint was selected reasonably rather than by a random fluctuation.

The error cases show that the remaining failures are not simply random mistakes. One example is a wireless driver discussion involving Broadcom, `iwlist scan`, device channels and Wi-Fi switches. The correct response asked the user to scan nearby devices and check whether Wi-Fi was disabled by a switch, but the model ranked a vague response, "could you point me to an example", higher. This suggests that when the context is short and contains domain-specific troubleshooting terms, the classifier may not fully understand which diagnostic action should follow. Another error occurs in a partition resizing conversation. The user discussed GParted, mounted partitions, a virtual machine and booting a live CD; the correct response explained how to boot the VM from a live CD, but the model selected "thanks!" as the best candidate. This is a typical hard-negative problem: a short conversational acknowledgement is common in dialogue data, but it is not a technically useful next response. A third example involves `xrandr` and VirtualBox display modes. The correct candidate mentioned reinstalling guest extensions, while the model selected "no idea." This reflects the limitation of the cross-encoder when the correct answer requires specific technical knowledge and the context is long.

These examples explain why Recall@5 is much higher than Recall@1. In many cases the model can place the correct response near the top, but the final first-rank decision can still be confused by generic replies, overlapping Ubuntu commands, or contexts where the important clue appears late and may be weakened by truncation.

Overall, the Part One results show that supervised fine-tuning is effective for adapting a base encoder to response selection. However, the system still has two important limitations. First, the negative candidates are fixed to nine distractors, so the task is easier than open retrieval over a very large response bank. Second, the model uses a maximum length of 256 tokens. This improves efficiency, but long Ubuntu conversations may contain earlier turns that are no longer visible to the model. A future improvement would be to compare this cross-encoder with a two-stage system: a bi-encoder for fast retrieval followed by a cross-encoder for re-ranking.

---

# Part Two：Document Summarization

## 1. Literature Review

Document summarization is used for news browsing, research reading and information management, with the aim of retaining the core facts of the original text and reducing reading costs. The three challenges of summarization are: long documents may exceed the model input limit; the generated results need to be faithful to the original text; and the model must select the most important content during compression. The base LLM summarization method usually requires fine-tuning on article-summary pairs. Its advantage is that it can be optimized for the dataset, but the cost is relatively high. Instruction-tuned LLMs can directly perform the summarization task based on prompts and are more suitable for this experiment.

Summarization methods can also be divided into extractive and abstractive approaches. Extractive systems select important sentences from the original document, which helps preserve factual consistency but may produce less fluent summaries. Abstractive systems generate new wording and can be more concise, but they may hallucinate or omit important details. The FLAN-T5 approach used here is abstractive, while the RAG component adds an extractive selection step before generation. Therefore, the final system combines sentence-level evidence selection with instruction-based generation.

This experiment uses `google/flan-t5-base` as the generative model and incorporates RAG within the document. RAG does not search for external knowledge but retrieves relevant chunks within the same article to alleviate the 512-token input limit.

## 2. Data and Preprocessing

The dataset is CNN/DailyMail. The `.story` files in `cnn_stories.tgz` and `dailymail_stories.tgz` are parsed. Each sample contains an article and reference highlights. The experiment first sampled 1,500 articles, constructed RAG prompts, filtered samples with more than 512 FLAN-T5 tokens, and finally retained 1,000 examples for the same evaluation of baseline and RAG.

Using the same final 1,000 examples for both methods is important for fair comparison. If baseline and RAG were evaluated on different subsets, the metric differences could be caused by sample difficulty rather than model design. The filtering criterion is only applied to the RAG prompt, because the baseline is explicitly defined as a first-512-token method. This means the baseline always fits the input window, while RAG examples are kept only when the retrieved two-chunk prompt can be used without truncation.

| Index       |        Value |
|-------------|-------------:|
| Sample size |         1000 |
| Average article length      | 684.21 words |
| Median article length      |    628 words |
| Average reference abstract length    |  49.11 words |
| Average compression ratio       |       0.0862 |
| Vocabulary list size       |        36295 |

## 3. Algorithm description and implementation

The baseline uses the first 512 tokens at the beginning of the article as the input to FLAN-T5. The RAG method first segments the article by sentence, constructing chunks of approximately 180 words with an overlap of about 40 words. The query is the first three sentences of the article; the retriever is `all-MiniLM-L6-v2`. The top-2 chunks are selected based on cosine similarity, concatenated in the original order, and then input into FLAN-T5.

The first three sentences are used as the retrieval query because news articles often introduce the main event, people and location at the beginning. This is not an oracle query, because it does not use the reference summary. The retrieved chunks are sorted back into their original article order before generation. This avoids feeding the generator a sequence that is ranked by similarity but narratively disordered. If an article contains only one chunk, that chunk is used directly.

```text
for each article:
    baseline_input = first_512_tokens(article)
    baseline_summary = FLAN_T5(prompt + baseline_input)

    query = first_three_sentences(article)
    chunks = sentence_aware_chunks(article)
    scores = cosine(embed(query), embed(chunk_i))
    retrieved = top2 chunks sorted by original position
    rag_summary = FLAN_T5(prompt + retrieved)
```

The generation parameters are `max_new_tokens=120`, `min_new_tokens=30`, `num_beams=4`, and `no_repeat_ngram_size=3`, without using random sampling. The evaluation metrics are ROUGE-1, ROUGE-2, ROUGE-L average F1 and corpus BLEU.

ROUGE and BLEU were used because they measure different aspects of overlap with the reference highlights. ROUGE-1 reflects unigram coverage, ROUGE-2 is stricter because it requires bigram overlap, and ROUGE-L captures longer ordered subsequences. BLEU is more precision-oriented and is less ideal for summarization than ROUGE, but it is useful as an additional indicator of phrase-level similarity.

## 4. Result Analysis

`figure 5：outputs/part_two/results/figures/rouge_comparison.png`  
`figure 6：outputs/part_two/results/figures/bleu_comparison.png`  
`figure 7：outputs/part_two/results/figures/length_statistics.png`  
`figure 8：outputs/part_two/results/figures/compression_ratio.png`

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---:|---:|---:|---:|
| FLAN-T5 baseline | 0.2613 | 0.0935 | 0.1878 | 0.1585 |
| FLAN-T5 + RAG | 0.2918 | 0.1125 | 0.2091 | 0.2079 |

RAG outperforms baseline in all metrics. The improvements in ROUGE-1 and ROUGE-L show that the retrieved fragments cover more reference summary information; the improvements in ROUGE-2 and BLEU indicate that phrase-level overlap is also better. The reason is that the baseline only relies on the beginning of the article, while RAG can select chunks that are more relevant to the main idea from the entire text. Since news texts often have an inverted pyramid structure and the beginning itself already contains a large amount of key information, the improvement in RAG is not extremely huge. However, the results still indicate that internal document retrieval can improve the summary quality of the input-constrained model.

The qualitative examples further support this interpretation. In one Daily Mail article about Mike Petrosino saying goodbye to his childhood dog before dying of cancer, the baseline summary focused narrowly on a quoted sentence about the dog, while the RAG summary correctly included the key facts that Petrosino was 21, had been diagnosed with cancer in eighth grade, and that doctors had said there was nothing more they could do. This is closer to the reference highlights, which emphasise the diagnosis, hospitalisation and final goodbye. In another article about Apple and e-book price fixing, the baseline captured the antitrust ruling, while the RAG summary added the refund amount and consumer impact. This shows that RAG can retrieve a chunk containing a concrete consequence of the event, not only the legal background. A third example is the Antarctic fishing boat fire. The RAG summary included the rescue of thirty-seven crew members and the severe burns suffered by two men, which directly matches the reference, whereas the baseline summary was more general.

At the same time, the examples also reveal why RAG should not be treated as a perfect solution. Some retrieved chunks contain repeated captions or byline information, and the generated summary may still omit secondary facts such as exact locations or dates. Therefore, the improvement comes mainly from better content selection under the input-length constraint, rather than from deeper factual reasoning.

However, the limitations of RAG also exist. The most obvious one is that if the query uses the first three sentences, it may lean towards the beginning of the article. Filtering RAG prompts with more than 512 tokens will cause the samples to slightly favor shorter or more compact articles. ROUGE and BLEU only measure lexical overlap and cannot fully evaluate factual consistency. Therefore, it is still necessary to conduct manual checks in combination with qualitative examples.

Another point is that CNN/DailyMail is a lead-biased dataset. Many news articles already place the most important information in the opening paragraphs, so a first-512-token baseline is strong. This makes the improvement from RAG harder to obtain and more meaningful when it appears. In datasets where key information is distributed more evenly, such as long reports or scientific papers, the benefit of retrieval-based chunk selection might be larger.

## Conclusion

Part One proved that the base MiniLM cross-encoder, after being fine-tuned with Ubuntu data, can effectively complete the 1-of-10 response ranking task. Part Two indicates that when the input length of FLAN-T5 is limited, intra-document RAG can improve the ROUGE and BLEU scores of the summary. Overall, the form of the task determines the model structure: response selection is more suitable for cross-encoder ranking, and document summarization is more suitable for an instruction-tuned generative model combined with retrievable content selection.
