#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusionkun.corruption import corrupt_target
from diffusionkun.dataset import Collator, TranslationJsonlDataset
from diffusionkun.model import DiffusionKun
from diffusionkun.sampling import diffuse_decode
from diffusionkun.utils import count_parameters, cosine_lr, ensure_dir, load_config, set_seed


def masked_ce_loss(logits, target, positions):
    if positions.sum().item() == 0:
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[positions], target[positions])


def decode_batch(sp, ids, pad_id):
    outs = []
    for row in ids.tolist():
        toks = [x for x in row if x != pad_id]
        outs.append(sp.decode(toks))
    return outs


@torch.no_grad()
def quick_eval(model, loader, sp, mask_id, pad_id, cfg, device, max_batches=2):
    model.eval()
    examples = []
    steps = cfg['diffusion']['decode_steps']
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        src_ids = batch.src_ids.to(device)
        src_mask = batch.src_mask.to(device)
        tgt_lengths = batch.tgt_lengths.to(device).clamp_max(cfg['model']['max_tgt_len'])
        y, _ = diffuse_decode(model, src_ids, src_mask, tgt_lengths, steps, mask_id, pad_id)
        hyps = decode_batch(sp, y.cpu(), pad_id)
        for s, r, h in zip(batch.raw_src, batch.raw_tgt, hyps):
            examples.append({'src': s, 'ref': r, 'hyp_oracle_len': h})
            if len(examples) >= 5:
                break
        if len(examples) >= 5:
            break
    model.train()
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--resume', default='')
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get('seed', 42)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bf16 = device.type == 'cuda' and cfg['training'].get('precision', 'bf16') == 'bf16'

    train_ds = TranslationJsonlDataset(
        path=cfg['paths']['train_jsonl'],
        spm_model=cfg['paths']['spm_model'],
        src_lang=cfg['data']['src_lang'],
        tgt_lang=cfg['data']['tgt_lang'],
        max_src_len=cfg['data']['max_src_len'],
        max_tgt_len=cfg['data']['max_tgt_len'],
    )
    dev_ds = TranslationJsonlDataset(
        path=cfg['paths']['dev_jsonl'],
        spm_model=cfg['paths']['spm_model'],
        src_lang=cfg['data']['src_lang'],
        tgt_lang=cfg['data']['tgt_lang'],
        max_src_len=cfg['data']['max_src_len'],
        max_tgt_len=cfg['data']['max_tgt_len'],
    )

    pad_id = train_ds.pad_id
    mask_id = train_ds.mask_id
    cfg['model']['vocab_size'] = train_ds.sp.get_piece_size()

    collator = Collator(pad_id=pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=min(16, cfg['training']['batch_size']),
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        collate_fn=collator,
    )

    model = DiffusionKun(
        vocab_size=cfg['model']['vocab_size'],
        d_model=cfg['model']['d_model'],
        encoder_layers=cfg['model']['encoder_layers'],
        decoder_layers=cfg['model']['decoder_layers'],
        heads=cfg['model']['heads'],
        ffn_dim=cfg['model']['ffn_dim'],
        dropout=cfg['model']['dropout'],
        max_src_len=cfg['model']['max_src_len'],
        max_tgt_len=cfg['model']['max_tgt_len'],
        pad_id=pad_id,
    ).to(device)

    print(f'Parameters: {count_parameters(model)/1e6:.1f}M')
    print(f'Vocab size: {cfg["model"]["vocab_size"]}, pad_id={pad_id}, mask_id={mask_id}')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        betas=(0.9, 0.98),
        weight_decay=cfg['training']['weight_decay'],
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt.get('step', 0)
        print(f'Resumed from {args.resume} at step {start_step}')

    out_dir = cfg['paths']['output_dir']
    ensure_dir(out_dir)
    with open(Path(out_dir) / 'config.resolved.json', 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    max_steps = cfg['training']['max_steps']
    accum = cfg['training']['grad_accum_steps']
    global_step = start_step
    model.train()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=max_steps, initial=start_step, dynamic_ncols=True)
    while global_step < max_steps:
        for batch in train_loader:
            # Update lr manually per optimizer step.
            lr = cosine_lr(
                global_step,
                max_steps,
                cfg['training']['warmup_steps'],
                cfg['training']['lr'],
                cfg['training']['min_lr'],
            )
            for group in optimizer.param_groups:
                group['lr'] = lr

            src_ids = batch.src_ids.to(device, non_blocking=True)
            src_mask = batch.src_mask.to(device, non_blocking=True)
            tgt_ids = batch.tgt_ids.to(device, non_blocking=True)
            tgt_lengths = batch.tgt_lengths.to(device, non_blocking=True).clamp_max(cfg['model']['max_tgt_len'])

            noisy_tgt, masked_pos, valid_pos, noise_ratio = corrupt_target(
                tgt_ids=tgt_ids,
                pad_id=pad_id,
                mask_id=mask_id,
                vocab_size=cfg['model']['vocab_size'],
                noise_min=cfg['diffusion']['noise_min'],
                noise_max=cfg['diffusion']['noise_max'],
                mask_replace_prob=cfg['diffusion']['mask_replace_prob'],
                random_replace_prob=cfg['diffusion']['random_replace_prob'],
            )

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                logits, length_logits = model(
                    src_ids=src_ids,
                    src_mask=src_mask,
                    noisy_tgt_ids=noisy_tgt,
                    tgt_valid_mask=valid_pos,
                    noise_ratio=noise_ratio,
                )
                masked_loss = masked_ce_loss(logits, tgt_ids, masked_pos)
                unmasked_loss = masked_ce_loss(logits, tgt_ids, valid_pos & ~masked_pos)
                length_loss = F.cross_entropy(length_logits, tgt_lengths)
                loss = (
                    cfg['loss']['masked_ce_weight'] * masked_loss
                    + cfg['loss']['unmasked_ce_weight'] * unmasked_loss
                    + cfg['loss']['length_loss_weight'] * length_loss
                )
                loss = loss / accum

            loss.backward()

            if (global_step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % cfg['training']['log_every'] == 0:
                pbar.set_description(
                    f"loss={loss.item()*accum:.3f} masked={masked_loss.item():.3f} len={length_loss.item():.3f} lr={lr:.2e}"
                )

            if global_step > 0 and global_step % cfg['training']['eval_every'] == 0:
                examples = quick_eval(model, dev_loader, train_ds.sp, mask_id, pad_id, cfg, device)
                print('\n[quick eval oracle length]')
                for ex in examples:
                    print('SRC:', ex['src'])
                    print('REF:', ex['ref'])
                    print('HYP:', ex['hyp_oracle_len'])
                    print('-' * 60)

            if global_step > 0 and global_step % cfg['training']['save_every'] == 0:
                ckpt_path = Path(out_dir) / f'step_{global_step}.pt'
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': global_step,
                    'config': cfg,
                    'pad_id': pad_id,
                    'mask_id': mask_id,
                }, ckpt_path)
                print(f'\nsaved {ckpt_path}')

            global_step += 1
            pbar.update(1)
            if global_step >= max_steps:
                break

    final_path = Path(out_dir) / 'final.pt'
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': global_step,
        'config': cfg,
        'pad_id': pad_id,
        'mask_id': mask_id,
    }, final_path)
    print(f'saved {final_path}')


if __name__ == '__main__':
    main()
