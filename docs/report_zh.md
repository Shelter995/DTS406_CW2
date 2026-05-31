# DTS406 Coursework 中文报告草稿

## 引言

本报告完成两个 NLP 任务：第一部分为对话系统中的回复选择，第二部分为文档摘要。回复选择属于检索/排序型任务，目标是在给定多轮对话上下文时，从候选回复中选出最合适的一条；文档摘要属于生成型任务，目标是将长文档压缩为简洁摘要。两部分分别使用独立数据集、模型和评价指标，以展示从数据预处理、模型实现到结果分析的完整流程。

---

# Part One：对话系统中的回复选择

## 1. 背景、挑战与方法选择

对话系统常见于客服机器人、技术支持助手和问答系统。与生成式对话不同，回复选择系统从预定义候选中排序，因此更强调相关性和稳定性。例如，电商客服可以从标准售后话术中选择答复，Ubuntu 技术支持助手可以根据用户错误描述检索已有解决方案。

回复选择有三个主要挑战：第一，多轮上下文中并非所有历史话语都同等重要，模型需要识别当前问题依赖的轮次；第二，难负样本可能与正确回复共享大量技术词汇，容易造成误判；第三，大规模候选排序需要兼顾效率和匹配精度。双编码器可以预计算候选向量，速度快但交互弱；交叉编码器将 `context` 和 `response` 一起输入模型，能捕捉 token-level 关系，但推理成本更高。

基础 LLM 方法可使用 BERT/MiniLM 类 encoder 进行任务微调；instruction-tuned LLM 方法可通过提示词让模型为候选回复打分。前者结构清楚、适合监督训练，后者理解能力强但速度慢、输出不稳定。本实验选择基础 MiniLM 交叉编码器作为主方法。

## 2. 数据与预处理

本实验使用 Ubuntu Dialogue Corpus v2.0。训练集为二分类格式 `Context, Utterance, Label`，验证和测试集为 1-of-10 ranking 格式，即 1 个真实回复和 9 个干扰回复。实验抽样 `train_100k`、`valid_5k` 和 `test_5k`。验证/测试候选顺序使用固定 seed 打乱，并保存正确回复索引。

模型输入保留 `__eou__` 和 `__eot__`，以保留话语和轮次边界。训练输入不删除停用词，因为否定词和功能词对语义匹配重要；统计分析阶段使用小写化、word tokenization 和 stopword removal。

| Split | 样本数 | 平均 context tokens | 平均 response/candidate tokens | 词汇表大小 |
|---|---:|---:|---:|---:|
| train_100k | 100000 | 75.07 | 14.94 | 114692 |
| valid_5k | 5000 | 80.24 | 17.23 | 27515 |
| test_5k | 5000 | 82.96 | 17.21 | 27886 |

## 3. 模型实现

模型为 `microsoft/MiniLM-L12-H384-uncased`。输入由 tokenizer 编码为 `[CLS] context [SEP] response [SEP]`，顶部加入二分类头，输出匹配/不匹配 logits。测试时，对同一 context 的 10 个候选分别计算 positive-class logit，并按分数排序。

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

训练设置为 3 epochs，batch size 32，learning rate 2e-5，warmup ratio 0.1，weight decay 0.01，优先使用 BF16。每轮在 valid_5k 上评估，并以 MRR 选择最佳 checkpoint。

## 4. 结果分析

`图 1：outputs/part_one/results/figures/validation_metrics.png`  
`图 2：outputs/part_one/results/figures/correct_rank_distribution.png`  
`图 3：outputs/part_one/results/figures/recall_at_k.png`  
`图 4：outputs/part_one/results/figures/score_gap_distribution.png`

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

结果显示模型在测试集上能较好排序候选回复。Recall@1 为 0.6336，说明约 63% 样本中正确回复排第一；Recall@5 达到 0.9124，说明大多数正确回复能进入前五。MRR 为 0.7492，说明整体排名靠前。错误主要来自语义相近的技术负样本和长上下文截断，例如多个候选都包含相同命令或错误信息时，模型可能过度依赖局部词面重合。

---

# Part Two：文档摘要

## 1. 背景、挑战与方法选择

文档摘要用于新闻浏览、研究阅读和信息管理，目标是保留原文核心事实并减少阅读成本。摘要任务的三个挑战是：长文档超过模型输入限制；生成结果需要忠实于原文；模型必须在压缩时选择最重要内容。基础 LLM 摘要方法通常需要在文章-摘要对上微调，优点是可针对数据集优化，但成本较高。instruction-tuned LLM 能直接根据提示词执行摘要任务，更适合本实验。

本实验使用 `google/flan-t5-base` 作为生成模型，并加入文档内部 RAG。RAG 不检索外部知识，而是在同一篇文章内检索相关 chunks，以缓解 512 token 输入限制。

## 2. 数据与预处理

数据集为 CNN/DailyMail，解析 `cnn_stories.tgz` 和 `dailymail_stories.tgz` 中的 `.story` 文件。每条样本包含 article 和 reference highlights。实验先抽样 1,500 篇，构建 RAG prompt 后过滤超过 512 FLAN-T5 tokens 的样本，最终保留 1,000 条用于 baseline 和 RAG 的相同评估。

| 指标 | 数值 |
|---|---:|
| 样本数 | 1000 |
| 平均文章长度 | 684.21 words |
| 中位文章长度 | 628 words |
| 平均参考摘要长度 | 49.11 words |
| 平均压缩比 | 0.0862 |
| 词汇表大小 | 36295 |

## 3. 摘要与 RAG 实现

Baseline 使用文章开头 512 tokens 输入 FLAN-T5。RAG 方法先按句子切分文章，构建约 180 words、重叠约 40 words 的 chunks；查询为文章前三句；检索器为 `all-MiniLM-L6-v2`；按余弦相似度选择 top-2 chunks，再按原文顺序拼接输入 FLAN-T5。

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

生成参数为 `max_new_tokens=120`、`min_new_tokens=30`、`num_beams=4`、`no_repeat_ngram_size=3`，不使用随机采样。评价指标为 ROUGE-1、ROUGE-2、ROUGE-L 平均 F1 和 corpus BLEU。

## 4. 结果分析

`图 5：outputs/part_two/results/figures/rouge_comparison.png`  
`图 6：outputs/part_two/results/figures/bleu_comparison.png`  
`图 7：outputs/part_two/results/figures/length_statistics.png`  
`图 8：outputs/part_two/results/figures/compression_ratio.png`

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---:|---:|---:|---:|
| FLAN-T5 baseline | 0.2613 | 0.0935 | 0.1878 | 0.1585 |
| FLAN-T5 + RAG | 0.2918 | 0.1125 | 0.2091 | 0.2079 |

RAG 在所有指标上高于 baseline。ROUGE-1 和 ROUGE-L 提升说明检索片段覆盖了更多参考摘要信息；ROUGE-2 和 BLEU 提升说明短语级重合也更好。其原因是 baseline 只依赖文章开头，而 RAG 能从全文选择与主旨更相关的 chunks。由于新闻文本常具有倒金字塔结构，开头本身已经包含大量关键信息，因此 RAG 的提升幅度不是极端巨大，但结果仍说明文档内部检索能改善输入受限模型的摘要质量。

局限性也存在。首先，query 使用前三句，可能偏向文章开头；其次，过滤超过 512 tokens 的 RAG prompt 会使样本略偏向较短或结构更紧凑的文章；最后，ROUGE 和 BLEU 只衡量词面重合，不能完全评价事实一致性，因此仍需要结合 qualitative examples 人工检查。

## 结论

Part One 证明了基础 MiniLM 交叉编码器经过 Ubuntu 数据微调后，能有效完成 1-of-10 回复排序。Part Two 表明，在 FLAN-T5 输入长度有限时，文档内部 RAG 能提升摘要的 ROUGE 和 BLEU。整体来看，任务形式决定了模型结构：回复选择更适合交叉编码排序，文档摘要更适合 instruction-tuned 生成模型结合检索式内容选择。

