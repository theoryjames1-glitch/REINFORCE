# 📘 RL Objectives for Language Models (with TRL functions)

This repo/demo shows how to implement **different RL training objectives** for language models using the **low-level functions from [Hugging Face TRL](https://github.com/huggingface/trl)**.

Instead of using `PPOTrainer` or `ReinforceTrainer`, we expose the **core math** directly in a single unified function.

---

## 🔑 What’s Inside

We provide a function `compute_rl_loss(...)` that supports:

* **Vanilla REINFORCE**
  Maximize reward-weighted log-prob.

* **KL-Regularized REINFORCE**
  Add a KL penalty to stay close to a reference model (RLHF-style).

* **PPO-Style Surrogate**
  Clipped ratio objective for stable updates.

* **Token-level REINFORCE**
  Support for token-wise rewards instead of one scalar per sequence.

* **Entropy Regularization**
  Encourage exploration by maximizing policy entropy.

---

## 🧩 Core TRL Utilities Used

* `logprobs_from_logits(logits, labels)` → token log-probs
* `whiten(values)` → advantage normalization (baseline subtraction)
* `entropy_from_logits(logits)` → entropy bonus
* `(logπ_current - logπ_ref)` → KL divergence

These are exactly what TRL’s trainers use under the hood.

---

## 🟢 Example: Unified Loss Function

```python
loss, stats = compute_rl_loss(
    model=model,
    enc_input_ids=enc.input_ids,
    prompt_len=len(tokenizer(prompt)["input_ids"]),
    rewards=[R],               # or per-token rewards
    mode="reinforce",          # "reinforce" | "kl" | "ppo" | "token"
    ref_model=ref_model,       # required for "kl"
    old_logprobs=old_logprobs, # required for "ppo"
    kl_coef=0.1,
    cliprange=0.2,
    ent_coef=0.01
)

loss.backward()
optimizer.step()
```

---

## 📊 Returned Stats

`compute_rl_loss` returns `(loss, stats)` where `stats` includes:

* `mean_reward`
* `mean_seq_logprob` or `mean_resp_logprob`
* `kl_mean` (if KL mode)
* `ratio_mean` (if PPO mode)
* `entropy` (if entropy reg used)

Useful for logging and debugging.

---

## 🚀 Workflow

1. **Generate response** with your LM (`model.generate`).
2. **Compute reward** (scalar or per-token).
3. **Call `compute_rl_loss`** with the right `mode`.
4. **Backprop and optimize** as usual.

---

## ✅ Why This Matters

* **Transparency** → see exactly what the loss is doing.
* **Flexibility** → swap objectives without rewriting your loop.
* **Consistency** → uses TRL’s core math functions, identical to what trainers do.

This README + function = a **theory of RL for language models, expressed in code**.

---

## 📌 Next Steps

* Add logging (TensorBoard/W\&B) to track rewards, KL, entropy.
* Experiment with mixing modes (e.g. KL-regularized PPO).
* Try per-token rewards from external models (e.g. toxicity classifiers, factuality checkers).


# PSEUDOCODE

## 🟢 Unified RL Loss Function (all modes)

```python
import torch
from trl.core import logprobs_from_logits, whiten, entropy_from_logits

def compute_rl_loss(
    model,
    enc_input_ids,
    prompt_len,
    rewards,
    mode="reinforce",        # "reinforce", "kl", "ppo", "token"
    ref_model=None,
    old_logprobs=None,
    kl_coef=0.1,
    cliprange=0.2,
    ent_coef=0.0,
    normalize_adv=True
):
    """
    Compute an RL loss using TRL core utilities.

    Modes:
      - "reinforce": vanilla REINFORCE
      - "kl": KL-regularized REINFORCE (needs ref_model)
      - "ppo": PPO-style clipped surrogate (needs old_logprobs)
      - "token": token-level REINFORCE (rewards per token)

    Args:
        model: Hugging Face CausalLM
        enc_input_ids: [batch, seq_len] token ids of prompt+response
        prompt_len: int, length of prompt tokens
        rewards: 
            - [batch] (sequence-level rewards)
            - or [batch, response_len] (token-level rewards if mode="token")
        mode: which RL loss to compute
        ref_model: reference model for KL (if mode="kl")
        old_logprobs: saved logprobs (if mode="ppo")
        kl_coef: KL penalty coefficient
        cliprange: PPO clip parameter
        ent_coef: entropy regularization coefficient
        normalize_adv: whiten sequence-level rewards

    Returns:
        loss: scalar tensor
        stats: dict of debug metrics
    """
    device = enc_input_ids.device
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)

    # Forward current model
    outputs = model(enc_input_ids)
    logits = outputs.logits  # [B, T, V]

    # Log-probs of chosen tokens
    log_probs = logprobs_from_logits(logits[:, :-1, :], enc_input_ids[:, 1:])

    # Mask for response tokens only
    mask = torch.zeros_like(log_probs, dtype=torch.bool)
    mask[:, prompt_len-1:] = 1

    # Default: sequence-level log-prob
    seq_logprob = (log_probs * mask).sum(dim=-1) / mask.sum(dim=-1)

    stats = {"mean_reward": rewards.mean().item()}

    # ---- Mode: Vanilla REINFORCE ----
    if mode == "reinforce":
        adv = whiten(rewards) if normalize_adv else rewards
        loss = -(adv * seq_logprob).mean()
        stats["mean_seq_logprob"] = seq_logprob.mean().item()

    # ---- Mode: KL-Regularized REINFORCE ----
    elif mode == "kl":
        assert ref_model is not None, "ref_model required for KL mode"
        with torch.no_grad():
            ref_logits = ref_model(enc_input_ids).logits
        logprobs_ref = logprobs_from_logits(ref_logits[:, :-1, :], enc_input_ids[:, 1:])
        kl_per_token = (log_probs - logprobs_ref) * mask
        kl_mean = kl_per_token.sum() / mask.sum()
        adv = whiten(rewards) if normalize_adv else rewards
        loss = -(adv * seq_logprob).mean() + kl_coef * kl_mean
        stats.update({
            "mean_seq_logprob": seq_logprob.mean().item(),
            "kl_mean": kl_mean.item()
        })

    # ---- Mode: PPO surrogate ----
    elif mode == "ppo":
        assert old_logprobs is not None, "old_logprobs required for PPO mode"
        adv = whiten(rewards) if normalize_adv else rewards
        ratio = torch.exp(seq_logprob - old_logprobs)
        loss1 = -adv * ratio
        loss2 = -adv * torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
        loss = torch.max(loss1, loss2).mean()
        stats.update({
            "ratio_mean": ratio.mean().item(),
            "mean_seq_logprob": seq_logprob.mean().item()
        })

    # ---- Mode: Token-level REINFORCE ----
    elif mode == "token":
        assert rewards.ndim == 2, "token-level rewards must be [batch, response_len]"
        resp_logprobs = log_probs * mask
        masked_rewards = rewards[:, :resp_logprobs.size(1)].to(device) * mask.float()
        loss = -(resp_logprobs * masked_rewards).sum() / mask.sum()
        stats["mean_resp_logprob"] = (resp_logprobs.sum(dim=-1) / mask.sum(dim=-1)).mean().item()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ---- Entropy regularization ----
    if ent_coef > 0.0:
        entropy = entropy_from_logits(logits).mean()
        loss -= ent_coef * entropy
        stats["entropy"] = entropy.item()

    return loss, stats
```

---

## 🟠 Example Usage

```python
full_text = prompt + generated_text
enc = tokenizer(full_text, return_tensors="pt").to("cuda")

# REINFORCE
loss, stats = compute_rl_loss(
    model=model,
    enc_input_ids=enc.input_ids,
    prompt_len=len(tokenizer(prompt)["input_ids"]),
    rewards=[R],
    mode="reinforce"
)

# KL-Regularized
loss, stats = compute_rl_loss(
    model=model,
    enc_input_ids=enc.input_ids,
    prompt_len=len(tokenizer(prompt)["input_ids"]),
    rewards=[R],
    mode="kl",
    ref_model=ref_model,
    kl_coef=0.1
)

# PPO surrogate
loss, stats = compute_rl_loss(
    model=model,
    enc_input_ids=enc.input_ids,
    prompt_len=len(tokenizer(prompt)["input_ids"]),
    rewards=[R],
    mode="ppo",
    old_logprobs=old_seq_logprobs,
    cliprange=0.2
)

# Token-level rewards
token_rewards = torch.ones((1, enc.input_ids.size(1) - prompt_len))  # dummy
loss, stats = compute_rl_loss(
    model=model,
    enc_input_ids=enc.input_ids,
    prompt_len=len(tokenizer(prompt)["input_ids"]),
    rewards=token_rewards,
    mode="token"
)
