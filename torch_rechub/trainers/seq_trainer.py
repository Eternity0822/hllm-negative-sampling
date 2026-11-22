"""Sequence Generation Model Trainer."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from ..basic.callback import EarlyStopper
from ..basic.loss_func import NCELoss


class SeqTrainer(object):
    """序列生成模型训练器.

    用于训练HSTU等序列生成模型。
    支持CrossEntropyLoss损失函数和生成式评估指标。

    Args:
        model (nn.Module): 要训练的模型
        optimizer_fn (torch.optim): 优化器函数，默认为torch.optim.Adam
        optimizer_params (dict): 优化器参数
        scheduler_fn (torch.optim.lr_scheduler): torch调度器类
        scheduler_params (dict): 调度器参数
        n_epoch (int): 训练轮数，默认10
        earlystop_patience (int): 早停耐心值，默认10
        device (str): 设备，'cpu'或'cuda'，默认'cpu'
        gpus (list): 多GPU的id列表，默认为[]
        model_path (str): 模型保存路径，默认为'./'
        loss_type (str): cross_entropy (default), legacy nce (full vocabulary),
            or sampled_softmax (positive plus sampled catalog negatives).
        loss_params (dict): For sampled_softmax: num_negatives=128 and optional
            negative_similarity_threshold for normalized HLLM item vectors.
        accumulation_steps (int): Positive number of microbatches per update, default 1.
        amp (bool): CUDA float16 autocast/scaling, disabled by default.
        max_grad_norm (float, optional): Clip unscaled gradients.
        eval_chunk_size (int): Positive candidate scoring chunk size, default 4096.

    Methods:
        fit: 训练模型
        evaluate: 评估模型
        predict: 生成预测

    Example:
        >>> trainer = SeqTrainer(
        ...     model=model,
        ...     optimizer_fn=torch.optim.Adam,
        ...     optimizer_params={'lr': 1e-3, 'weight_decay': 1e-5},
        ...     device='cuda'
        ... )
        >>> trainer.fit(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader
        ... )
    """

    def __init__(
        self,
        model,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device='cpu',
        gpus=None,
        model_path='./',
        loss_type='cross_entropy',
        loss_params=None,
        model_logger=None,
        accumulation_steps=1,
        amp=False,
        max_grad_norm=None,
        eval_chunk_size=4096,
        q_table=None,
        in_batch_neg=False,
        use_logq=False,
        curriculum_epoch=None,
    ):
        self.model = model  # for uniform weights save method in one gpu or multi gpu
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.accumulation_steps = accumulation_steps
        self.amp = bool(amp and self.device.type == 'cuda')
        # Small-temperature cosine scores amplify gradients; a modest initial
        # scale avoids skipping all updates in short tutorial runs.
        # 只有 self.amp=True（且设备是 CUDA）时才真正使用 GradScaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp, init_scale=1024)
        self.max_grad_norm = max_grad_norm
        self.eval_chunk_size = eval_chunk_size
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **(scheduler_params or {}))

        # 损失函数
        if loss_type == 'sampled_softmax':
            params = dict(loss_params or {})
            self.num_negatives = params.pop('num_negatives', 128)
            self.negative_similarity_threshold = params.pop('negative_similarity_threshold', None)
            if self.negative_similarity_threshold is not None:
                if not 0 <= self.negative_similarity_threshold <= 1 or not hasattr(self.model, 'get_item_embeddings'):
                    raise ValueError("negative_similarity_threshold requires normalized get_item_embeddings() and a value in [0, 1]")
            if params:
                raise ValueError("Unknown sampled_softmax parameters: " + ", ".join(params))
            if len(gpus) > 1:
                raise ValueError("sampled_softmax currently supports one device; use the separate multi-GPU branch")
            if not all(hasattr(self.model, method) for method in ('encode', 'score_items', 'vocab_size')):
                raise ValueError("sampled_softmax requires encode(), score_items() and vocab_size")
            if self.model.vocab_size < 3:
                raise ValueError("sampled_softmax requires at least two real items plus PAD")
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        elif loss_type == 'nce':
            if loss_params is None:
                loss_params = {"temperature": 0.1, "ignore_index": 0}
            self.loss_fn = NCELoss(**loss_params)
        elif loss_type == 'cross_entropy':
            if loss_params is None:
                loss_params = {"ignore_index": 0}
            self.loss_fn = nn.CrossEntropyLoss(**loss_params)
        else:
            raise ValueError("loss_type must be cross_entropy, nce or sampled_softmax")

        self.loss_type = loss_type

        # ---- 负采样变体，仅对 sampled_softmax 生效 ----
        self.q_table = None # 统计热门程度
        self.q_cdf = None # q_table 的 cumsum
        if q_table is not None:
            q = torch.as_tensor(q_table, dtype=torch.float32)
            if q.numel() != self.model.vocab_size or bool((q < 0).any()) or abs(q.sum().item() - 1.0) > 1e-3:
                raise ValueError("q_table must be a probability vector of length vocab_size")
            self.q_table = q.to(self.device)
            self.q_cdf = q.cumsum(0).to(self.device)
            self.q_cdf[-1] = 1.0
        if use_logq and self.q_table is None:
            raise ValueError("use_logq requires q_table")
        self.in_batch_neg = bool(in_batch_neg)
        self.use_logq = bool(use_logq)
        self.curriculum_epoch = curriculum_epoch

        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path
        self.model_logger = model_logger

    def fit(self, train_dataloader, val_dataloader=None):
        """训练模型.

        Args:
            train_dataloader (DataLoader): 训练数据加载器
            val_dataloader (DataLoader): 验证数据加载器

        Returns:
            dict: 训练历史
        """
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        os.makedirs(self.model_path, exist_ok=True)
        model_file = os.path.join(self.model_path, "model.pth")

        for logger in self._iter_loggers():
            logger.log_hyperparams({'n_epoch': self.n_epoch, 'learning_rate': self.optimizer.param_groups[0]['lr'], 'loss_type': self.loss_type})

        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            # 训练阶段
            train_loss = self.train_one_epoch(train_dataloader, epoch=epoch_i)
            history['train_loss'].append(train_loss)

            # Collect metrics
            logs = {'train/loss': train_loss, 'learning_rate': self.optimizer.param_groups[0]['lr']}

            # 验证阶段
            if val_dataloader is not None:
                val_loss, val_accuracy = self.evaluate(val_dataloader) # 在验证集上评估
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                logs['val/loss'] = val_loss
                logs['val/accuracy'] = val_accuracy
                logs['auc'] = val_accuracy  # For compatibility with EarlyStopper

                print(f"epoch: {epoch_i}, validation: loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
                improved = val_accuracy > self.early_stopper.best_auc
                # EarlyStopper owns the only best snapshot. Keep it on CPU so
                # a frozen HLLM item table does not occupy GPU memory twice.
                weights = self.model.state_dict()
                if improved:
                    weights = {name: value.cpu() for name, value in weights.items()}
                stop_training = self.early_stopper.stop_training(val_accuracy, weights)
                if improved:
                    # Save before logging or the next epoch: an interruption
                    # must not discard a completed validation's best model.
                    torch.save(self.early_stopper.best_weights, model_file)

                if stop_training:
                    print(f'validation: best accuracy: {self.early_stopper.best_auc}')
                    break

            for logger in self._iter_loggers(): # 默认为空
                logger.log_metrics(logs, step=epoch_i)
            print("Current lr : {}".format(self.optimizer.param_groups[0]['lr']))
            if self.scheduler is not None:
                self.scheduler.step()

        if val_dataloader is not None and self.early_stopper.best_weights is not None:
            self.model.load_state_dict(self.early_stopper.best_weights)
        torch.save(self.model.state_dict(), model_file)

        for logger in self._iter_loggers():
            logger.finish()

        return history

    def _iter_loggers(self):
        """Return logger instances as a list.

        Returns
        -------
        list
            Active logger instances. Empty when ``model_logger`` is ``None``.
        """
        if self.model_logger is None:
            return []
        if isinstance(self.model_logger, (list, tuple)):
            return list(self.model_logger)
        return [self.model_logger]

    def _compute_next_token_loss(self, logits, seq_tokens, targets):
        """Cross-entropy over the whole sequence (autoregressive next-token).

        ``logits[:, i, :]`` predicts the following real token; the final
        non-PAD position predicts the held-out target. This works with left
        and right padding (not just the final physical array slot). PAD
        positions are ignored based on the *current* token. This matters for
        left-padded sequences: the last PAD slot may have the first real item as
        its shifted label, but the model should not learn a PAD -> item
        transition.
        """
        vocab_size = logits.size(-1)
        next_tokens = self._next_tokens(seq_tokens, targets) # [B, L]
        if vocab_size > 0:
            logits = logits.clone()
            logits[..., 0] = torch.finfo(logits.dtype).min
        return self.loss_fn(logits.reshape(-1, vocab_size), next_tokens.reshape(-1)) # 将 logits 展平为 [B * L, V] 将 next_tokens 展平为 [B * L]

    def train_one_epoch(self, data_loader, epoch, log_interval=10):
        """Train the model for a single epoch.

        Args:
            data_loader (DataLoader): Training data loader.
            log_interval (int): Interval (in steps) for logging average loss.

        Returns:
            float: Average training loss for this epoch.
        """
        self.model.train()
        total_loss = 0
        epoch_loss = 0
        batch_count = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        self.optimizer.zero_grad()
        for i, (seq_tokens, _seq_positions, seq_time_diffs, targets) in enumerate(tk0):
            # Move tensors to the target device. ``seq_positions`` is unused by
            # ``HSTUModel`` (which derives positions internally); keep the
            # dataloader API stable but drop it here.
            seq_tokens = seq_tokens.to(self.device) # [B, L]
            seq_time_diffs = seq_time_diffs.to(self.device) # [B, L]
            targets = targets.to(self.device).reshape(-1) # [B]

            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                # 负例课程：前 curriculum_epoch 个 epoch 走全词表，之后切采样
                use_full_vocab = (self.loss_type == 'sampled_softmax'
                                  and self.curriculum_epoch is not None
                                  and epoch < self.curriculum_epoch)
                if self.loss_type == 'sampled_softmax' and not use_full_vocab:
                    loss = self._sampled_loss(self.model.encode(seq_tokens, seq_time_diffs), seq_tokens, targets)
                else: # 前 curriculum_epoch 个 epoch 走这里
                    logits = self.model(seq_tokens, seq_time_diffs) # [B, L, V]
                    loss = self._compute_next_token_loss(logits, seq_tokens, targets)
            
            # Scale by the actual microbatch count, including the short tail.
            group_start = (i // self.accumulation_steps) * self.accumulation_steps
            group_size = min(self.accumulation_steps, len(data_loader) - group_start)
            self.scaler.scale(loss / group_size).backward()
            if (i + 1) % self.accumulation_steps == 0 or i + 1 == len(data_loader):
                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

        return epoch_loss / batch_count if batch_count > 0 else 0

    def evaluate(self, data_loader):
        """Evaluate the model on a validation/test data loader.

        Loss matches training: full-vocabulary CE or sampled candidate CE.
        Sampled loss uses fixed validation draws and is not full-vocabulary CE.
        accuracy is top-1 hit on the held-out next item only (useful as a
        ranking-style proxy metric).

        Returns:
            tuple: ``(avg_loss, top1_accuracy)``.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        # Fixed validation candidates make loss comparisons reproducible and
        # do not consume the training sampler's random stream.
        generator = torch.Generator(device=self.device).manual_seed(0)

        with torch.no_grad():
            for seq_tokens, _seq_positions, seq_time_diffs, targets in tqdm.tqdm(data_loader, desc="evaluating", smoothing=0, mininterval=1.0):
                seq_tokens = seq_tokens.to(self.device)
                seq_time_diffs = seq_time_diffs.to(self.device)
                targets = targets.to(self.device).reshape(-1)

                if self.loss_type == 'sampled_softmax':
                    states = self.model.encode(seq_tokens, seq_time_diffs)
                    loss = self._sampled_loss(states, seq_tokens, targets, generator=generator)
                    last = self._last_positions(seq_tokens)
                    last_states = states[torch.arange(states.size(0), device=self.device), last]
                    predictions = self._top1(last_states)
                else:
                    logits = self.model(seq_tokens, seq_time_diffs)
                    loss = self._compute_next_token_loss(logits, seq_tokens, targets)
                    last_logits = logits[torch.arange(logits.size(0), device=self.device), self._last_positions(seq_tokens)].clone()
                    last_logits[:, 0] = torch.finfo(last_logits.dtype).min
                    predictions = torch.argmax(last_logits, dim=-1)
                total_loss += loss.item()
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.numel()

        avg_loss = total_loss / max(1, len(data_loader))
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy

    def _last_positions(self, seq_tokens):
        valid = seq_tokens.ne(0)
        if not valid.any(dim=1).all():
            raise ValueError("Every sequence must contain at least one non-PAD item")
        positions = torch.arange(seq_tokens.size(1), device=seq_tokens.device).expand_as(seq_tokens)
        last = positions.masked_fill(~valid, -1).max(dim=1).values
        return last

    def _next_tokens(self, seq_tokens, targets):
        if targets.numel() != seq_tokens.size(0) or targets.le(0).any():
            raise ValueError("Provide one non-PAD target per history")
        last = self._last_positions(seq_tokens)
        next_tokens = torch.cat([seq_tokens[:, 1:], torch.zeros_like(seq_tokens[:, :1])], dim=1)
        next_tokens.scatter_(1, last.unsqueeze(1), targets.reshape(-1, 1))
        return next_tokens.masked_fill(seq_tokens.eq(0), 0)

    def _sampled_loss(self, states, seq_tokens, targets, generator=None):
        labels = self._next_tokens(seq_tokens, targets) # [B, L]
        valid = labels.ne(0)
        positive_ids = labels[valid] # 把 [32, 50] 中所有 True 位置的标量取出来，压成一维 [N_valid]
        states = states[valid] # states 原本 [B, L, D]，按 [B, L] 的 bool 掩码索引前两维，得到 [N_valid, D]
        n_states = states.size(0)

        # 负例预算分配：两种来源都开时各占一半
        # k_ib: batch 内负采样的样本数 k_q: 流行度反变换采样的样本数 
        k_ib = k_q = 0
        if self.in_batch_neg and self.q_table is not None:
            k_ib = self.num_negatives // 2
            k_q = self.num_negatives - k_ib
        elif self.in_batch_neg:
            k_ib = self.num_negatives
        elif self.q_table is not None:
            k_q = self.num_negatives
        
        neg_list, logq_list = [], []
        if k_ib > 0:
            # M1/M3：in-batch 负例（本 batch 其他位置的 positive）
            # randperm 返回一个 0, 1, ..., n_states - 1 的一个随机排列
            perm = torch.randperm(n_states, device=self.device, generator=generator)
            cand = positive_ids[perm[:min(k_ib, n_states)]] # 从正例池里取出 min(k_ib, n_states) 个 id 作为 in-batch 负例（不重复）
            neg_list.append(cand)
            logq_list.append(torch.zeros(cand.shape, dtype=torch.float32, device=self.device)) # 不使用 logq
        if k_q > 0:
            # M2/M3：按流行度 q 反变换采样 + logQ 修正
            u = torch.rand(k_q, device=self.device, generator=generator) # 取 k_q 个 0~1 的数
            # 逆变换采样：u 均匀时，落进区间 0~cdf[k] 的概率 = 区间长度 = q_table[k]
            # 即流行度越高，越容易被选为负样本
            q_ids = torch.searchsorted(self.q_cdf, u).clamp(min=1) # 对 u 里的每一个 u[i] 返回 sorted_seq 中第一个 >= u[i] 的位置，即物品 id 这个物品被选中的概率正好是 q_table[id]
            neg_list.append(q_ids)
            logq_list.append(torch.log(self.q_table[q_ids].clamp_min(1e-12)))
        if not neg_list:
            # 回退：均匀随机（= B2 基线行为）可能抽到重复的 id
            neg_list.append(torch.randint(1, self.model.vocab_size, (self.num_negatives,), device=self.device, generator=generator))
            logq_list.append(torch.zeros(self.num_negatives, dtype=torch.float32, device=self.device))
        negatives = torch.cat(neg_list)
        logq = torch.cat(logq_list)
        
        # paired = True 只计算 states 与 positive_ids 对应的 embedding 相乘求和
        positive_scores = self.model.score_items(states, positive_ids, paired=True).unsqueeze(1) # [N_valid, 1] 每个 query 对正样本的打分
        # 这里 paired = False 因为每个 query 要和所有 128 个负样本打分
        negative_scores = self.model.score_items(states, negatives) # [N_valid, num_negatives]
        # positive_ids[:, None] 和 negatives[None, :] 将二者广播成相同形状，相同的地方置为极小值
        negative_scores = negative_scores.masked_fill(positive_ids[:, None].eq(negatives[None, :]), torch.finfo(negative_scores.dtype).min)
        if self.negative_similarity_threshold is not None: # 0.99
            with torch.no_grad():
                # [N_valid, num_negatives] 每个正样本与负样本的相似度，太高的要过滤掉
                similarity = self.model.get_item_embeddings(positive_ids) @ self.model.get_item_embeddings(negatives).t()
            negative_scores = negative_scores.masked_fill(similarity > self.negative_similarity_threshold, torch.finfo(negative_scores.dtype).min)
        if self.use_logq:
            negative_scores = negative_scores + logq.unsqueeze(0) # logq 修正
        # cat 后形状为 [N_valid, 1 + num_negatives] 每行对应一个样本的 1 正（第 0 列） + 128 负（第 1~128 列）的分类问题
        # cross_entropy 第二个参数放正样本在 logits 类别维里的下标，所以通常把正样本放第 0 列，label 全 0，最简单
        return F.cross_entropy(torch.cat([positive_scores, negative_scores], dim=1).float(), torch.zeros(states.size(0), dtype=torch.long, device=self.device))

    def _top1(self, states):
        best_scores = torch.full((states.size(0),), float('-inf'), device=self.device)
        best_ids = torch.ones(states.size(0), dtype=torch.long, device=self.device)
        for start in range(1, self.model.vocab_size, self.eval_chunk_size):
            item_ids = torch.arange(start, min(start + self.eval_chunk_size, self.model.vocab_size), device=self.device)
            scores, indices = self.model.score_items(states, item_ids).max(dim=-1)
            improved = scores > best_scores
            best_scores = torch.where(improved, scores, best_scores)
            best_ids = torch.where(improved, item_ids[indices], best_ids)
        return best_ids

    @torch.no_grad()
    def predict(self, data_loader, top_k=10, filter_seen=False):
        """Return exact catalog top-k IDs with bounded scoring memory.

        top_k is a positive integer. PAD is always excluded. History filtering is opt-in because a repeated
        interaction may be the true next item. This is not sampled evaluation.
        """
        model = self.model.module if hasattr(self.model, 'module') else self.model
        top_k = min(top_k, model.vocab_size - 1)
        was_training = self.model.training
        self.model.eval()
        predictions = []
        try:
            for seq_tokens, _positions, time_diffs, _targets in data_loader:
                seq_tokens, time_diffs = seq_tokens.to(self.device), time_diffs.to(self.device)
                states = model.encode(seq_tokens, time_diffs) # [B, L, D]
                # 从每个样本的整条序列隐向量 [B, L, D] 里，挑出该样本最后一个非 PAD 位置对应的那个
                # 物品的 D 维向量作为表征向量，得到 [B, D]
                states = states[torch.arange(states.size(0), device=self.device), self._last_positions(seq_tokens)] # [B, D]
                # best_scores[b, j]：第 b 个样本第 j 个候选的分数
                best_scores = torch.empty(states.size(0), 0, device=self.device) # [B, 0]
                # best_ids[b, j]：对应的 item id
                best_ids = torch.empty(states.size(0), 0, device=self.device, dtype=torch.long) # [B, 0] 
                # 分块计算过程：开始时 best_scores [B, 0]; best_ids [B, 0]
                #   计算第一块：调用 model.score_items 对块内 chunksize 个物品打分，得到 [B, chunksize]
                #              接到 best_scores 上：[B, chunksize] 对应 best_ids: [B, chunksize]
                #              之后选出 topk 分数重新赋值给 best_scores 以及 best_ids, 所以每块计算完后，best_scores 和 best_ids 的形状为 [B, topk]
                #   计算第二块：对块内 chunksize 个物品打分，得到 [B, chunksize]
                #              接到 best_scores 上：[B, topk + chunksize] best_ids: [B, topk + chunksize] 
                #              之后选出 topk 分数重新赋值给 best_scores 以及 best_ids 形状为 [B, topk]
                #   以此类推...
                for start in range(1, model.vocab_size, self.eval_chunk_size): #  分块对物品打分，0 是 PAD 从 1 开始
                    # ids 范围: [1, 4096] [4097, 电影总数]
                    ids = torch.arange(start, min(start + self.eval_chunk_size, model.vocab_size), device=self.device)
                    # scores[b, c] = 第 b 个样本的 query 与第 c 个候选 item 的匹配分数
                    scores = model.score_items(states, ids) # [B, 块内物品数量] 
                    if filter_seen:
                        seen = torch.zeros(states.size(0), ids.size(0), dtype=torch.bool, device=self.device)
                        # Mark history IDs only within the current catalog chunk.
                        rows, positions = torch.where((seq_tokens >= start) & (seq_tokens < start + ids.size(0)))
                        seen[rows, seq_tokens[rows, positions] - start] = True
                        scores = scores.masked_fill(seen, float('-inf'))
                    scores = torch.cat([best_scores, scores], dim=-1)
                    candidates = torch.cat([best_ids, ids.expand(states.size(0), -1)], dim=-1)
                    best_scores, positions = scores.topk(min(top_k, scores.size(1)), dim=-1) # [B, topk] [B, topk]
                    best_ids = candidates.gather(1, positions) # [B, topk]
                # Not enough eligible items: -1, never a seen item pretending
                # to be a legal recommendation.
                best_ids = best_ids.masked_fill(~torch.isfinite(best_scores), -1) # [B, topk]
                predictions.append(best_ids.cpu()) # predictions 列表里每个元素形状 [B, top_k]
        finally:
            self.model.train(was_training)
        # cat 之后形状为 [data_loader 样本总数, topk(id)]
        return torch.cat(predictions) if predictions else torch.empty(0, top_k, dtype=torch.long)

    def export_onnx(self, output_path, batch_size=2, seq_length=50, vocab_size=None, opset_version=14, dynamic_batch=True, device=None, verbose=False, onnx_export_kwargs=None):
        """Export the trained sequence generation model to ONNX format.

        This method exports sequence generation models (e.g., HSTU) to ONNX format.
        Unlike other trainers, sequence models use positional arguments (seq_tokens, seq_time_diffs)
        instead of dict input, making ONNX export more straightforward.

        Args:
            output_path (str): Path to save the ONNX model file.
            batch_size (int): Batch size for dummy input (default: 2).
            seq_length (int): Sequence length for dummy input (default: 50).
            vocab_size (int, optional): Vocabulary size for generating dummy tokens.
                If None, will try to get from model.vocab_size.
            opset_version (int): ONNX opset version (default: 14).
            dynamic_batch (bool): Enable dynamic batch size (default: True).
            device (str, optional): Device for export ('cpu', 'cuda', etc.).
                If None, defaults to 'cpu' for maximum compatibility.
            verbose (bool): Print export details (default: False).
            onnx_export_kwargs (dict, optional): Extra kwargs forwarded to ``torch.onnx.export``.

        Returns:
            bool: True if export succeeded, False otherwise.

        Example:
            >>> trainer = SeqTrainer(hstu_model, ...)
            >>> trainer.fit(train_dl, val_dl)
            >>> trainer.export_onnx("hstu.onnx", vocab_size=10000)

            >>> # Export on specific device
            >>> trainer.export_onnx("hstu.onnx", vocab_size=10000, device="cpu")
        """
        import warnings

        # Use provided device or default to 'cpu'
        export_device = device if device is not None else 'cpu'

        # Handle DataParallel wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model
        original_device = next(model.parameters()).device
        original_training = model.training

        # Get vocab_size from model if not provided
        if vocab_size is None:
            if hasattr(model, 'vocab_size'):
                vocab_size = model.vocab_size
            elif hasattr(model, 'item_num'):
                vocab_size = model.item_num
            else:
                raise ValueError("vocab_size must be provided or model must have 'vocab_size' or 'item_num' attribute")

        # Generate dummy inputs on the export device
        dummy_seq_tokens = torch.randint(0, vocab_size, (batch_size, seq_length), device=export_device)
        dummy_seq_time_diffs = torch.zeros(batch_size, seq_length, dtype=torch.float32, device=export_device)

        # Configure dynamic axes
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {"seq_tokens": {0: "batch_size", 1: "seq_length"}, "seq_time_diffs": {0: "batch_size", 1: "seq_length"}, "output": {0: "batch_size", 1: "seq_length"}}

        # Ensure output directory exists
        import os
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            model.to(export_device)
            model.eval()
            with torch.no_grad():
                import inspect

                export_kwargs = {
                    "f": output_path,
                    "input_names": ["seq_tokens",
                                    "seq_time_diffs"],
                    "output_names": ["output"],
                    "dynamic_axes": dynamic_axes,
                    "opset_version": opset_version,
                    "do_constant_folding": True,
                    "verbose": verbose,
                }

                if onnx_export_kwargs:
                    overlap = set(export_kwargs.keys()) & set(onnx_export_kwargs.keys())
                    overlap.discard("dynamo")
                    if overlap:
                        raise ValueError("onnx_export_kwargs contains keys that overlap with explicit args: "
                                         f"{sorted(overlap)}. Please set them via export_onnx() parameters instead.")
                    export_kwargs.update(onnx_export_kwargs)

                # Auto-pick exporter:
                # - dynamic_axes present => prefer legacy exporter (dynamo=False) for dynamic batch/seq
                # - otherwise prefer dynamo exporter (dynamo=True) on newer torch
                sig = inspect.signature(torch.onnx.export)
                if "dynamo" in sig.parameters:
                    if "dynamo" not in export_kwargs:
                        export_kwargs["dynamo"] = False if dynamic_axes is not None else True
                else:
                    export_kwargs.pop("dynamo", None)

                torch.onnx.export(model, (dummy_seq_tokens, dummy_seq_time_diffs), **export_kwargs)

            if verbose:
                print(f"Successfully exported ONNX model to: {output_path}")
                print("  Input names: ['seq_tokens', 'seq_time_diffs']")
                print(f"  Vocab size: {vocab_size}")
                print(f"  Opset version: {opset_version}")
                print(f"  Dynamic batch: {dynamic_batch}")

            return True

        except Exception as e:
            warnings.warn(f"ONNX export failed: {str(e)}")
            raise RuntimeError(f"Failed to export ONNX model: {str(e)}") from e
        finally:
            # Export is an inspection step, not a permanent device/mode switch.
            model.to(original_device)
            model.train(original_training)

    def visualization(self, seq_length=50, vocab_size=None, batch_size=2, depth=3, show_shapes=True, expand_nested=True, save_path=None, graph_name="model", device=None, dpi=300, **kwargs):
        """Draw the sequence model with generated token and time inputs.

        vocab_size defaults to model.vocab_size or model.item_num.
        Other options are shared with utils.visualization.visualize_model.
        Example: trainer.visualization(seq_length=20, save_path="model.png").
        """
        from ..utils.visualization import visualize_model

        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        if vocab_size is None:
            vocab_size = getattr(model, 'vocab_size', getattr(model, 'item_num', None))
        if vocab_size is None or vocab_size < 2:
            raise ValueError("Provide vocab_size >= 2, or define model.vocab_size/model.item_num.")

        # Token 0 is padding; use real item IDs so the graph follows a valid sequence.
        tokens = torch.randint(1, vocab_size, (batch_size, seq_length))
        time_diffs = torch.zeros(batch_size, seq_length)
        return visualize_model(model, input_data=(tokens, time_diffs), depth=depth, show_shapes=show_shapes, expand_nested=expand_nested, save_path=save_path, graph_name=graph_name, device=device, dpi=dpi, **kwargs)
