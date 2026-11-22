# HLLM 负采样优化（torch-rechub）

基于 [Datawhale 的 torch-rechub](https://github.com/datawhalechina/torch-rechub) 开源项目中 HLLM 生成式推荐的训练侧改动：将均匀随机负例替换为流行度对齐的难负例采样，并引入 logQ 采样偏置修正与两阶段课程训练。不改模型结构、不新增参数，全部改动集中在训练侧。

## 项目介绍

HLLM（Hierarchical Large Language Model for Recommendation，[ByteDance 原始工作](https://github.com/bytedance/HLLM)）是生成式推荐框架：物品侧使用离线冻结的 LLM（TinyLlama-1.1B）文本 embedding，用户侧训练 2 层 causal Transformer，打分 = L2 归一化 cosine 相似度 × 温度缩放（1/0.07）。

**基线的问题**：采样版训练（`sampled_softmax`）对全目录**均匀随机**采负例。在温度 0.07 下，随机物品的 logit 远低于正例，CE 梯度对它们几乎为零，负例预算被"过易"的样本浪费；而评估时的全目录 top-k 榜单里真正挤掉正例的主要是**流行物品**，均匀采样让这一区间严重欠训练。

**本仓库的改动**（三个可开关的组件）：

1. **in-batch 负例**（M1）：取本 batch 其他位置的 positive 作为负例，提供上下文相关的难负例；
2. **流行度对齐采样 + logQ 修正**（M2）：按 q_i ∝ (freq_i+1)^0.5 采样目录负例（freq 仅统计自训练集，无泄漏），并对采到的负例 logit 加 log q_j 消除采样偏置（word2vec sampled softmax 技巧）；
3. **负例课程**：前 `--curriculum_epoch` 个 epoch 用全词表 CE 保证覆盖，之后切采样负例目标，把后期梯度集中在 top-k 决策边界。

M3 = M1 + M2 组合，用于消融（结果见下）。

## 实验结果

数据集：MovieLens-1M（leave-last-out 切分）；`--num_negatives 128`、`--curriculum_epoch 3`、seed=2022。算力：Kaggle 免费加速器 **Tesla T4**（共 14.56GB 可用显存）。

| 实验 | 负例策略 | Hit@10 | Hit@50 | Hit@200 | NDCG@10 | NDCG@50 | NDCG@200 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 全词表 NCE（上限参照） | 0.0305 | 0.1228 | 0.3291 | 0.0136 | 0.0333 | 0.0641 |
| B2 | 均匀随机（基线） | 0.0209 | 0.0921 | 0.2773 | 0.0102 | 0.0252 | 0.0526 |
| M1 | in-batch | 0.0257 | 0.0871 | 0.2482 | 0.0136 | 0.0268 | 0.0505 |
| M2 | 流行度 q + logQ | 0.0255 | 0.1073 | 0.2861 | 0.0120 | 0.0293 | 0.0558 |
| M3 | in-batch + 流行度 q + logQ | 0.0147 | 0.0530 | 0.1632 | 0.0067 | 0.0148 | 0.0310 |

- **M2 全面超过均匀基线 B2**：Hit@10 +22.0%、NDCG@10 +17.6%、Hit@50 +16.5%，六项指标全部不降；每步打分成本约为全词表 B1 的 1/29，整体排序质量达到 B1 的 ~84%~88%。
- **M1 头部大涨（Hit@10 +23%、NDCG@10 +33%）但深位略降**：in-batch 负例硬度高（上下文相关），但缺少全局流行结构覆盖。
- **M3 全面垫底**：两种负例来源简单叠加会因流行区覆盖重复、in-batch 侧缺失 logQ 修正而过度压制流行物品（各 k 均匀下跌 ~41%），说明两种负例来源不可加。

## 复现教程

### 0. 环境

Python ≥ 3.9，PyTorch（CUDA 版）。

```bash
git clone https://github.com/Eternity0822/hllm-negative-sampling.git
cd hllm-negative-sampling
pip install -r requirements.txt
```

### 1. 序列数据预处理（自动下载 MovieLens-1M）

```bash
python examples/generative/data/ml-1m/preprocess_ml_hstu.py
```

输出 `examples/generative/data/ml-1m/processed/` 下的 `vocab.pkl`、`train/val/test_data.pkl`。

### 2. 生成 item embeddings（自动下载 TinyLlama-1.1B，约 2.4GB）

```bash
python examples/generative/data/ml-1m/preprocess_hllm_data.py --model_type tinyllama --device cuda
```

输出 `item_embeddings_tinyllama.pt`。

### 3. 训练与评估（五个实验）

以下公共配置均为脚本默认值，显式列出以便核对：`--model_type tinyllama --epoch 10 --batch_size 64 --learning_rate 1e-3 --max_seq_len 200 --device cuda --seed 2022 --num_negatives 128`。

注意每个实验使用不同的 `--save_dir`，避免覆盖同一个 `model.pth`。

```bash
# B1：全词表 NCE（默认 loss）
python examples/generative/run_hllm_movielens.py \
    --loss_type nce --save_dir outputs/B1

# B2：均匀随机负例（基线）
python examples/generative/run_hllm_movielens.py \
    --loss_type sampled_softmax --num_negatives 128 --save_dir outputs/B2

# M1：in-batch 负例 + 课程
python examples/generative/run_hllm_movielens.py \
    --loss_type sampled_softmax --num_negatives 128 \
    --in_batch_neg --curriculum_epoch 3 --save_dir outputs/M1

# M2：流行度 q 采样 + logQ + 课程
python examples/generative/run_hllm_movielens.py \
    --loss_type sampled_softmax --num_negatives 128 \
    --q_sample --logq --curriculum_epoch 3 --save_dir outputs/M2

# M3：两者组合
python examples/generative/run_hllm_movielens.py \
    --loss_type sampled_softmax --num_negatives 128 \
    --in_batch_neg --q_sample --logq --curriculum_epoch 3 --save_dir outputs/M3
```

每次运行结束会自动打印 test 集的 HR/NDCG@10/50/200（全目录精确 top-k 检索，排除 PAD）。上表数字即 seed=2022 单次运行的结果；如需更稳的数字，请用多个 seed 取均值。

## 改动说明

相对 torch-rechub 原仓库，本仓库只保留 HLLM 相关的最小文件集，并修改了两个文件：

- `torch_rechub/trainers/seq_trainer.py`
  - `SeqTrainer.__init__` 新增 `q_table / in_batch_neg / use_logq / curriculum_epoch` 参数，校验并缓存 q 分布及其 CDF；
  - `train_one_epoch` 新增课程开关：前 `curriculum_epoch` 个 epoch 走全词表 CE，之后切采样负例目标；
  - `_sampled_loss` 将均匀随机采样替换为「in-batch 负例 + 流行度逆变换采样 + logQ 修正」，保留原有的自冲突 mask 与 cosine 假负例过滤。
- `examples/generative/run_hllm_movielens.py`
  - 用训练集统计流行度分布 q（freq 仅来自 train，防止泄漏）；
  - 新增 CLI 参数 `--curriculum_epoch / --in_batch_neg / --q_sample / --logq / --neg_alpha`。

## 致谢

- Datawhale [torch-rechub](https://github.com/datawhalechina/torch-rechub)：HLLM 基线、SeqTrainer 与数据预处理管线（本仓库在其上修改）
- ByteDance [HLLM](https://github.com/bytedance/HLLM)：生成式推荐原始方法
- [MovieLens-1M](https://grouplens.org/datasets/movielens/)：实验数据集