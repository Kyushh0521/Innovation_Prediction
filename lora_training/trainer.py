import os
import json
import math
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import gc
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

from utils import now_str, setup_logging
from dataset import SFTDataset, get_dataloader
from adv import project_onto_l2_ball, grad_norm


def train(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(cfg.get("result_output"), now_str())
    setup_logging(out_dir)

    model_out_dir = cfg.get("model_output", os.path.join(out_dir, "model"))
    os.makedirs(model_out_dir, exist_ok=True)

    logging.info("=== 训练：SFT + 动态对抗训练 + LoRA ===")
    logging.info(json.dumps(cfg, indent=2, ensure_ascii=False))

    model_name = cfg.get("model_name_or_path")
    cache_dir = cfg.get("model_cache", None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir=cache_dir, low_cpu_mem_usage=True, attn_implementation="sdpa")
    # 可添加 attn_implementation="flash_attention_2" 参数加速训练
    # model.resize_token_embeddings(len(tokenizer))

    # 从顶层配置读取 LoRA 参数
    r_val = cfg.get("r", 8)
    alpha_val = cfg.get("lora_alpha", 16)
    dropout_val = cfg.get("dropout", 0.0)
    target_modules = cfg.get("target_modules", ["q_proj", "v_proj"])
    peft_cfg = LoraConfig(
        r=r_val,
        lora_alpha=alpha_val,
        lora_dropout=dropout_val,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.config.use_cache = False
    # 关闭 reentrant，以支持 torch.autograd.grad
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_cfg)
    model.to(device)

    # 定义验证集评估函数
    @torch.no_grad()
    def evaluate_loss(model, dataloader, device):
        """计算验证集上的平均 CrossEntropy Loss"""
        model.eval()  # 切换到评估模式
        total_loss_val = 0.0
        total_steps = 0
        
        for batch in tqdm(dataloader, desc="正在评估验证集", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss_val += outputs.loss.item()
            total_steps += 1
        
        model.train()  # 切换回训练模式
        return total_loss_val / (total_steps + 1e-12)
    
    ds = SFTDataset(cfg.get("train_dataset"), tokenizer, cfg)
    dl = get_dataloader(ds, tokenizer, cfg)

    # 加载验证集
    eval_path = cfg.get("eval_dataset", None)
    eval_dl = None
    if eval_path and os.path.exists(eval_path):
        eval_ds = SFTDataset(eval_path, tokenizer, cfg)
        eval_dl = get_dataloader(eval_ds, tokenizer, cfg, shuffle=False)
    else:
        logging.warning("未配置 eval_dataset 或文件不存在，将跳过最佳模型保存功能。")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    # 自动计算总步数
    accum_steps = int(cfg.get("gradient_accumulation_steps", 1))
    steps_per_epoch = len(dl) // accum_steps
    T = steps_per_epoch * cfg.get("epochs", 3)
    warmup_steps = int(cfg.get("warmup_frac", 0.0) * T)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=T,
    )

    # 读取对抗训练参数
    eps_max = cfg.get("eps_max", 0.00001)
    alpha = cfg.get("adv_alpha", 0.3)
    K = cfg.get("steps", 3)
    lambda_adv = cfg.get("lambda_adv", 1.0)

    eps_probe = cfg.get("eps_probe", 1e-2)      # 探测噪声半径
    probe_per_steps = cfg.get("probe_per_steps", 2) # 探测频率
    ema_beta = cfg.get("ema_beta", 0.9)          # EMA 平滑系数
    ema_grad_adv = None                          # 对抗梯度范数 EMA

    # 动态权重 λt 参数
    lambda_min = cfg.get("lambda_min", 0.3)
    lambda_max = cfg.get("lambda_max", 2.0)

    iter_count = 0
    global_step = 0

    twarmup_steps = int(cfg.get("twarmup_frac", 0.1) * T)
    twarmup_steps = max(twarmup_steps, 1)

    hist = {"step": [], "loss": [], "ce": [], "adv": [], "gamma": [], "eps_t": [], "lambda_t": [], "lr": [], "val_loss": [], "val_step": []}
    best_val_loss = float('inf') # 初始化最佳 Loss

    logging_steps = cfg.get("logging_steps", None)

    pbar = tqdm(total=T, desc="训练开始")
    model.train()
    optimizer.zero_grad()

    for epoch in range(cfg.get("epochs", 5)):
        for batch in dl:
            if global_step >= T:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            B = input_ids.size(0)

            # 获取 Embedding 并计算动态扰动参数 (Epsilon)
            embed_layer = model.get_input_embeddings() #type: ignore
            embeds = embed_layer(input_ids).detach()
            embeds.requires_grad_(False)
            del input_ids             # 转换后立即释放

            # 1. 计算标准生成任务 (CE) 梯度并获取实时范数
            out_ce = model(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)
            ce_loss = out_ce.loss
            ce_loss_val = ce_loss.item()
            
            # 使用 retain_graph=True 以支持后续可能的计算
            ce_grads = torch.autograd.grad(ce_loss, trainable, retain_graph=True, allow_unused=True)
            
            # 实时计算当前 batch 的 CE 梯度范数
            current_ce_sq = 0.0
            scale = 1.0 / accum_steps
            with torch.no_grad():
                for p, g in zip(trainable, ce_grads):
                    if g is not None:
                        g_dt = g.detach()
                        current_ce_sq += g_dt.norm(2).pow(2).item()
                        if p.grad is None: p.grad = g_dt * scale
                        else: p.grad.add_(g_dt, alpha=scale)
            grad_ce = math.sqrt(current_ce_sq + 1e-30)
            del ce_grads, out_ce, ce_loss

            if lambda_adv > 0.0:
                if iter_count % probe_per_steps == 0:
                    # 生成随机探测噪声
                    delta_probe = torch.randn_like(embeds)
                    flat_p = delta_probe.view(B, -1)
                    delta_probe = (delta_probe / (flat_p.norm(p=2, dim=1, keepdim=True).view(B, 1, 1) + 1e-12)) * eps_probe
                    
                    probe_inputs = (embeds + delta_probe.detach()).requires_grad_(True)
                    out_probe = model(inputs_embeds=probe_inputs, attention_mask=attention_mask, labels=labels)
                    
                    # 计算探测梯度范数
                    p_grads = torch.autograd.grad(out_probe.loss, trainable, retain_graph=False, allow_unused=True)
                    p_sq = sum(g.detach().norm(2).pow(2).item() for g in p_grads if g is not None)
                    grad_adv_probed = math.sqrt(p_sq + 1e-30)
                    
                    # 更新 EMA
                    if ema_grad_adv is None: ema_grad_adv = grad_adv_probed
                    else: ema_grad_adv = ema_beta * ema_grad_adv + (1 - ema_beta) * grad_adv_probed
                    del p_grads, out_probe, delta_probe

                # 计算动态参数
                gamma_t = grad_ce / (ema_grad_adv + 1e-12) if ema_grad_adv else 1.0
                if global_step < twarmup_steps:
                    eps_t = eps_max * (global_step / twarmup_steps)
                else:
                    eps_t = eps_max * math.tanh(gamma_t) # 使用 tanh 映射
                eps_t = float(max(min(eps_t, eps_max), 1e-6))
                lambda_t = float(np.clip(1.0 / (gamma_t + 1e-12), lambda_min, lambda_max))
            else:
                # 纯 LoRA 模式：强制置零，防止日志出现异常大数值
                gamma_t = 0.0
                eps_t = 0.0
                lambda_t = 0.0


            adv_loss_val = 0.0  # 初始化默认值，防止日志报错
            if lambda_adv > 0.0:
                delta = torch.zeros_like(embeds)
                max_delta = torch.zeros_like(embeds)
                max_adv_loss_scalar = -1e12

                for k in range(K):
                    delta.requires_grad_()
                    out_adv_k = model(inputs_embeds=embeds + delta, attention_mask=attention_mask, labels=labels)
                    adv_loss = out_adv_k.loss
                    if adv_loss.item() > max_adv_loss_scalar:
                        max_adv_loss_scalar = adv_loss.item()
                        max_delta = delta.detach().clone()

                    grad_delta = torch.autograd.grad(adv_loss, delta, retain_graph=False)[0]
                    gn = grad_delta.view(B, -1).norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                    delta = (delta + alpha * grad_delta / gn.view(B, 1, 1)).detach()
                    delta = project_onto_l2_ball(delta, eps_t)

                out_final = model(inputs_embeds=embeds + max_delta, attention_mask=attention_mask, labels=labels)
                adv_loss_final = out_final.loss
                adv_loss_val = adv_loss_final.item()
                
                adv_grads = torch.autograd.grad(adv_loss_final, trainable, retain_graph=False, allow_unused=True)
                adv_scale = lambda_t / accum_steps
                with torch.no_grad():
                    for p, g in zip(trainable, adv_grads):
                        if g is not None:
                            g_dt = g.detach()
                            if p.grad is None: p.grad = g_dt * adv_scale
                            else: p.grad.add_(g_dt, alpha=adv_scale)
                del adv_grads, out_final, delta, max_delta

            del embeds, attention_mask, labels
       
            # 优化器更新与日志 (Step)
            iter_count += 1
            if iter_count % accum_steps == 0:
                # 梯度裁剪与更新
                torch.nn.utils.clip_grad_norm_(trainable, cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                pbar.update(1)

                # 每隔一定步数清理一次碎片
                if global_step % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                # 记录日志
                current_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else 0.0
                total_loss_display = ce_loss_val + lambda_t * adv_loss_val
                
                hist["step"].append(global_step)
                hist["loss"].append(total_loss_display)
                hist["ce"].append(ce_loss_val)
                hist["adv"].append(adv_loss_val)
                hist["gamma"].append(gamma_t)
                hist["eps_t"].append(eps_t)
                hist["lambda_t"].append(lambda_t)
                hist["lr"].append(current_lr)

                if logging_steps and (global_step % logging_steps == 0):
                    logging.info(f"步数={global_step} 总损失={total_loss_display:.4f} 标准损失={ce_loss_val:.4f} 对抗损失={adv_loss_val:.4f} 扰动半径={eps_t:.4f} 梯度敏感因子={gamma_t:.4f} 对抗权重={lambda_t:.4f} 学习率={current_lr:.6f}")

                # 验证与保存
                eval_steps = cfg.get("eval_steps", 100)
                if eval_dl and (global_step % eval_steps == 0):
                    val_loss = evaluate_loss(model, eval_dl, device)
                    logging.info(f"步数={global_step} 当前验证损失={val_loss:.6f} 最低验证损失={best_val_loss:.6f}")
                    hist["val_loss"].append(val_loss)
                    hist["val_step"].append(global_step)

                    if val_loss < best_val_loss:
                        logging.info(f"🚀 新最佳模型: {val_loss:.6f}")
                        best_val_loss = val_loss
                        best_dir = os.path.join(model_out_dir, "best")
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        
                if global_step % cfg.get("save_steps", 100) == 0:
                    ckpt = os.path.join(model_out_dir, f"ckpt-{global_step}")
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)

    # 核心训练循环修改结束
    if (iter_count % accum_steps) != 0:
        optimizer.zero_grad()
        logging.info("训练结束：已丢弃未满一个累积步的梯度。")

    pbar.close()

    steps = hist["step"]

    plt.figure()
    plt.plot(steps, hist["loss"], label="total")
    plt.plot(steps, hist["ce"], label="ce")
    plt.plot(steps, hist["adv"], label="adv")
    # 如果存在验证集评估点，则用虚线/标记绘制验证损失
    if len(hist.get("val_step", [])) > 0:
        plt.plot(hist["val_step"], hist["val_loss"], label="val", linestyle="--", marker="o")
    plt.legend()
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    
    if lambda_adv > 0.0:
        plt.figure()
        plt.plot(steps, hist["gamma"], label="gamma_t")
        plt.plot(steps, hist["eps_t"], label="eps_t")
        plt.legend()
        plt.xlabel("Training Steps")
        plt.ylabel("Value")
        plt.title("Gamma and Eps over Training")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "gamma_eps_curve.png"))

    #  保存详细的训练日志
    train_keys = ["step", "loss", "ce", "adv", "gamma", "eps_t", "lr"]
    train_hist = [dict(zip(train_keys, vals)) for vals in zip(*(hist[k] for k in train_keys))]
    with open(os.path.join(out_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(train_hist, f, indent=2)

    # 保存验证日志
    if len(hist["val_step"]) > 0:
        val_keys = ["val_step", "val_loss"]
        val_hist = [dict(zip(val_keys, vals)) for vals in zip(*(hist[k] for k in val_keys))]
        with open(os.path.join(out_dir, "eval_history.json"), "w", encoding="utf-8") as f:
            json.dump(val_hist, f, indent=2)

    final_dir = os.path.join(model_out_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    logging.info("训练完成。")
    logging.info(f"最终模型已保存至 {final_dir}")
    logging.info(f"结果输出目录 = {out_dir}")