# train.py (resume from specified epoch 20; save every 10 epochs)
import os
import glob
import re
import time
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# external modules - make sure these exist
from dataset import SEGAN_Dataset
from hparams import hparams
from model import Generator, Discriminator

# optional: for saving audio samples
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except Exception:
    SOUNDFILE_AVAILABLE = False

# ------------------------
# Helpers
# ------------------------
def safe_tensor_stats(tensor):
    if tensor is None:
        return "None"
    try:
        return f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}"
    except Exception:
        return str(tensor.shape)

def save_unified_checkpoint(save_dir, epoch, generator, discriminator, g_optimizer, d_optimizer):
    """
    Save a unified checkpoint containing model weights and optimizer states.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        cp = {
            'epoch': epoch,
            'g_state_dict': generator.state_dict(),
            'd_state_dict': discriminator.state_dict(),
            'g_optimizer': g_optimizer.state_dict(),
            'd_optimizer': d_optimizer.state_dict()
        }
        p = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save(cp, p)
        print(f"[SAVE] Unified checkpoint saved: {p}")
    except Exception as e:
        print("[WARN] Failed to save unified checkpoint:", e)

def save_model_state_only(save_dir, epoch, generator, discriminator, suffix=""):
    """
    Save separate model state_dicts (legacy style).
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        gp = os.path.join(save_dir, f"G_epoch{epoch}{suffix}.pkl")
        dp = os.path.join(save_dir, f"D_epoch{epoch}{suffix}.pkl")
        torch.save(generator.state_dict(), gp)
        torch.save(discriminator.state_dict(), dp)
        print(f"[SAVE] Model-only states saved: {gp}, {dp}")
    except Exception as e:
        print("[WARN] Failed to save model-only states:", e)

def save_debug_checkpoint(save_dir, epoch, generator, discriminator, g_optimizer, d_optimizer, suffix="debug"):
    try:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(generator.state_dict(), os.path.join(save_dir, f"G_epoch{epoch}_{suffix}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_dir, f"D_epoch{epoch}_{suffix}.pth"))
        ckpt = {
            'epoch': epoch,
            'g_state_dict': generator.state_dict(),
            'd_state_dict': discriminator.state_dict(),
            'g_optimizer': g_optimizer.state_dict(),
            'd_optimizer': d_optimizer.state_dict()
        }
        torch.save(ckpt, os.path.join(save_dir, f"checkpoint_epoch{epoch}_{suffix}.pth"))
        print(f"[DEBUG SAVE] debug checkpoint saved epoch {epoch} suffix={suffix}")
    except Exception as e:
        print("[WARN] failed to save debug checkpoint:", e)

# ------------------------
# Resume helpers
# ------------------------
def find_latest_unified_ckpt(save_path):
    pattern = os.path.join(save_path, "checkpoint_epoch*.pth")
    files = glob.glob(pattern)
    if not files:
        return None, None
    max_epoch = -1
    chosen = None
    for f in files:
        m = re.search(r"checkpoint_epoch(\d+)\.pth", os.path.basename(f))
        if m:
            e = int(m.group(1))
            if e > max_epoch:
                max_epoch = e
                chosen = f
    return max_epoch, chosen

def find_latest_g_pkl(save_path):
    pattern = os.path.join(save_path, "G_epoch*.pkl")
    files = glob.glob(pattern)
    if not files:
        return None, None
    max_epoch = -1
    chosen = None
    for f in files:
        m = re.search(r"G_epoch(\d+)", os.path.basename(f))
        if m:
            e = int(m.group(1))
            if e > max_epoch:
                max_epoch = e
                chosen = f
    return max_epoch, chosen

def find_matching_d_for_epoch(save_path, epoch):
    unified = os.path.join(save_path, f"checkpoint_epoch{epoch}.pth")
    if os.path.exists(unified):
        return unified
    pattern = os.path.join(save_path, f"D_epoch{epoch}*.pkl")
    files = glob.glob(pattern)
    return files[0] if files else None

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # load hyperparams (this prints config)
    para = hparams()
    # compatibility defaults
    if not hasattr(para, "g_update_freq"):
        para.g_update_freq = 5
    if not hasattr(para, "num_workers"):
        para.num_workers = 4
    if not hasattr(para, "win_len"):
        para.win_len = getattr(para, "win_len", 16384)

    # build models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=para.lr_G, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=para.lr_D, betas=(0.5, 0.999))

    # dataloader
    dataloader = DataLoader(
        SEGAN_Dataset(para),
        batch_size=para.batch_size,
        shuffle=True,
        num_workers=para.num_workers,
        pin_memory=True
    )

    # prepare fixed reference batch (as before)
    dataset = dataloader.dataset
    try:
        ref_batch = dataset.ref_batch(para.ref_batch_size).to(device)
    except Exception as e:
        print("[WARN] dataset.ref_batch failed:", e)
        sample_clean, sample_noisy = dataset[0]
        ref_batch = torch.cat([sample_clean.unsqueeze(0), sample_noisy.unsqueeze(0)], dim=1).to(device)
        if ref_batch.size(0) == 1 and para.ref_batch_size > 1:
            ref_batch = ref_batch.repeat(para.ref_batch_size, 1, 1)

    # ------------------------
    # Resume logic: use specified epoch 20 files if present
    # ------------------------
    start_epoch = 20  # force start from epoch 20 (0-indexed)
    optimizers_restored = False

    # user-specified checkpoints (from your message)
    save_path = para.save_path
    given_g_name = "G_epoch20_cond0.0322.pkl"
    given_d_name = "D_epoch20_loss0.3197.pkl"
    g_full = os.path.join(save_path, given_g_name)
    d_full = os.path.join(save_path, given_d_name)

    # Try to load provided G first
    if os.path.exists(g_full):
        try:
            generator.load_state_dict(torch.load(g_full, map_location=device))
            print(f"[RESUME] Loaded generator from provided file: {g_full}")
        except Exception as e:
            print("[RESUME] Failed to load provided G file:", e)
            # fallback to automatic search
            g_epoch, g_path = find_latest_g_pkl(save_path)
            if g_path:
                try:
                    generator.load_state_dict(torch.load(g_path, map_location=device))
                    start_epoch = g_epoch
                    print(f"[RESUME] Fallback loaded G: {g_path} (epoch {g_epoch})")
                except Exception as e2:
                    print("[RESUME] Fallback G load failed:", e2)
            else:
                print("[RESUME] No G checkpoint found; starting from scratch.")
    else:
        print(f"[RESUME] Provided G checkpoint not found at {g_full}. Searching for latest G in {save_path} ...")
        g_epoch, g_path = find_latest_g_pkl(save_path)
        if g_path:
            try:
                generator.load_state_dict(torch.load(g_path, map_location=device))
                start_epoch = g_epoch
                print(f"[RESUME] Loaded latest G: {g_path} (epoch {g_epoch})")
            except Exception as e:
                print("[RESUME] Failed to load latest G:", e)
        else:
            print("[RESUME] No G checkpoint found; starting from scratch.")

    # Try to load provided D
    if os.path.exists(d_full):
        try:
            raw = torch.load(d_full, map_location=device)
            state = raw['d_state_dict'] if isinstance(raw, dict) and 'd_state_dict' in raw else raw
            try:
                discriminator.load_state_dict(state, strict=True)
                print(f"[RESUME] Discriminator fully loaded (strict) from provided file: {d_full}")
            except Exception as e_strict:
                print(f"[RESUME] strict load failed for provided D: {e_strict}")
                try:
                    res = discriminator.load_state_dict(state, strict=False)
                    if isinstance(res, tuple) or isinstance(res, list):
                        missing_keys, unexpected_keys = res
                    elif isinstance(res, dict):
                        missing_keys = res.get('missing_keys', [])
                        unexpected_keys = res.get('unexpected_keys', [])
                    else:
                        missing_keys, unexpected_keys = [], []
                    print(f"[RESUME] Discriminator loaded with strict=False from provided D. missing_keys: {missing_keys}; unexpected_keys: {unexpected_keys}")
                except Exception as e_ns:
                    print("[RESUME] Non-strict load failed for provided D:", e_ns)
                    # try dummy forward
                    try:
                        print("[RESUME] Attempting dummy forward for provided D...")
                        dummy_x = torch.zeros(1, 2, para.win_len, device=device)
                        dummy_ref = dummy_x.clone()
                        with torch.no_grad():
                            _ = discriminator(dummy_x, dummy_ref)
                        res2 = discriminator.load_state_dict(state, strict=False)
                        print(f"[RESUME] After dummy forward, loaded provided D (non-strict). Result: {res2}")
                    except Exception as e_dummy:
                        print("[RESUME] Dummy-forward for provided D failed:", e_dummy)
        except Exception as e:
            print("[RESUME] Failed to load provided D file:", e)
    else:
        print(f"[RESUME] Provided D checkpoint not found at {d_full}. Searching for matching D_epoch{start_epoch} in {save_path} ...")
        # fallback: find matching D for start_epoch
        d_path = find_matching_d_for_epoch(save_path, start_epoch)
        if d_path:
            try:
                print(f"[RESUME] Found candidate D checkpoint: {d_path}")
                raw = torch.load(d_path, map_location=device)
                state = raw['d_state_dict'] if isinstance(raw, dict) and 'd_state_dict' in raw else raw
                try:
                    discriminator.load_state_dict(state, strict=True)
                    print("[RESUME] discriminator loaded with strict=True from fallback")
                except Exception as e_strict:
                    print("[RESUME] strict load failed for discriminator (fallback):", e_strict)
                    try:
                        res = discriminator.load_state_dict(state, strict=False)
                        if isinstance(res, tuple) or isinstance(res, list):
                            missing_keys, unexpected_keys = res
                        elif isinstance(res, dict):
                            missing_keys = res.get('missing_keys', [])
                            unexpected_keys = res.get('unexpected_keys', [])
                        else:
                            missing_keys, unexpected_keys = [], []
                        print(f"[RESUME] discriminator loaded with strict=False (fallback). missing_keys: {missing_keys}; unexpected_keys: {unexpected_keys}")
                    except Exception as e_ns:
                        print("[RESUME] Non-strict load also failed (fallback):", e_ns)
                        try:
                            print("[RESUME] Attempting dummy forward (fallback) to init dynamic layers...")
                            dummy_x = torch.zeros(1, 2, para.win_len, device=device)
                            dummy_ref = dummy_x.clone()
                            with torch.no_grad():
                                _ = discriminator(dummy_x, dummy_ref)
                            res2 = discriminator.load_state_dict(state, strict=False)
                            print(f"[RESUME] After dummy forward (fallback), load_state_dict result: {res2}")
                        except Exception as e_dummy:
                            print("[RESUME] Dummy-forward (fallback) failed:", e_dummy)
            except Exception as e:
                print("[RESUME] Failed to load D fallback:", e)
        else:
            print("[RESUME] No matching D checkpoint found for epoch", start_epoch, "; continuing with current D initialization.")

    # We did not load unified optimizer states (unless the loaded checkpoint contained them and load succeeded)
    print(f"[RESUME] start_epoch set to {start_epoch} (0-indexed). optimizers_restored={optimizers_restored}")
    print("[RESUME] Training loop will use: for epoch in range(start_epoch, para.n_epoch):")

    # ------------------------
    # Training loop
    # ------------------------
    avg_d_history = []
    avg_g_history = []
    avg_gcond_history = []
    avg_l1_history = []
    avg_dreal_history = []
    avg_dfake_history = []

    def try_free_cuda():
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # save interval
    SAVE_EVERY = 10  # only save every 10 epochs

    try:
        for epoch in range(start_epoch, para.n_epoch):
            print("\n" + "=" * 50)
            print(f"Epoch {epoch + 1}/{para.n_epoch}")
            print("=" * 50)

            epoch_g_loss_updates = 0.0
            epoch_g_cond_updates = 0.0
            epoch_g_updates = 0
            epoch_d_loss = 0.0
            epoch_l1 = 0.0
            d_real_sum = 0.0
            d_fake_sum = 0.0
            steps = 0

            for batch_idx, (batch_clean, batch_noisy) in enumerate(dataloader):
                steps += 1
                batch_clean = batch_clean.to(device)
                batch_noisy = batch_noisy.to(device)

                batch_z = torch.randn(
                    batch_clean.size(0),
                    para.size_z[0],
                    para.size_z[1],
                    device=device
                )

                # Train D
                discriminator.zero_grad()
                try:
                    real_batch = torch.cat([batch_clean, batch_noisy], dim=1)
                    real_outputs = discriminator(real_batch, ref_batch)
                    real_labels = torch.ones_like(real_outputs) * 0.9
                    clean_loss = torch.mean((real_outputs - real_labels) ** 2)

                    with torch.no_grad():
                        generated = generator(batch_noisy, batch_z)

                    fake_batch = torch.cat([generated, batch_noisy], dim=1)
                    fake_outputs = discriminator(fake_batch, ref_batch)
                    fake_labels = torch.zeros_like(fake_outputs) + 0.1
                    noisy_loss = torch.mean((fake_outputs - fake_labels) ** 2)

                    d_loss = clean_loss + noisy_loss

                    if torch.isnan(d_loss) or torch.isinf(d_loss):
                        print("[ERROR] d_loss NaN/Inf at epoch", epoch, "batch", batch_idx)
                        raise RuntimeError("d_loss NaN/Inf")

                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    d_optimizer.step()
                except RuntimeError as re:
                    if 'out of memory' in str(re):
                        print("[ERROR] CUDA OOM at epoch", epoch, "batch", batch_idx)
                        try_free_cuda()
                        save_debug_checkpoint(para.save_path, epoch, generator, discriminator, g_optimizer, d_optimizer, suffix=f"oom_b{batch_idx}")
                        raise
                    else:
                        raise

                # Train G (every g_update_freq batches)
                do_update_g = ((batch_idx % para.g_update_freq) == 0)
                if do_update_g:
                    generator.zero_grad()
                    generated = generator(batch_noisy, batch_z)
                    fake_batch = torch.cat([generated, batch_noisy], dim=1)
                    outputs = discriminator(fake_batch, ref_batch)

                    g_adv_loss = para.w_g_loss1 * torch.mean((outputs - 1.0) ** 2)
                    l1_dist = torch.abs(generated - batch_clean)
                    g_cond_loss = para.w_g_loss2 * torch.mean(l1_dist)
                    g_loss = g_adv_loss + g_cond_loss

                    if torch.isnan(g_loss) or torch.isinf(g_loss):
                        print("[ERROR] g_loss NaN/Inf at epoch", epoch, "batch", batch_idx)
                        raise RuntimeError("g_loss NaN/Inf")

                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    g_optimizer.step()

                    epoch_g_loss_updates += g_loss.item()
                    epoch_g_cond_updates += g_cond_loss.item()
                    epoch_g_updates += 1
                else:
                    g_loss = torch.tensor(float('nan'))
                    g_cond_loss = torch.tensor(float('nan'))

                # accumulate stats
                epoch_d_loss += d_loss.item()
                epoch_l1 += torch.mean(torch.abs(generated - batch_clean)).item()
                d_real_sum += real_outputs.mean().item()
                d_fake_sum += fake_outputs.mean().item()

                # periodic logging
                if batch_idx % 100 == 0:
                    cur_l1 = torch.mean(torch.abs(generated - batch_clean)).item()
                    print(f"Batch {batch_idx:4d}/{len(dataloader)} | "
                          f"D_loss: {d_loss.item():.4f} | "
                          f"G_loss: {g_loss.item() if not torch.isnan(g_loss) else float('nan'):.4f} | "
                          f"G_cond: {g_cond_loss.item() if not torch.isnan(g_cond_loss) else float('nan'):.4f}")
                    print(f"          实际L1: {cur_l1:.4f} | "
                          f"D_real: {real_outputs.mean().item():.3f} | "
                          f"D_fake: {fake_outputs.mean().item():.3f}")

                    if SOUNDFILE_AVAILABLE:
                        try:
                            os.makedirs(os.path.join(para.save_path, "samples"), exist_ok=True)
                            gen_np = generated[0].detach().cpu().numpy().reshape(-1)
                            clean_np = batch_clean[0].detach().cpu().numpy().reshape(-1)
                            sf.write(os.path.join(para.save_path, f"samples/gen_e{epoch+1}_b{batch_idx}.wav"), gen_np, para.fs)
                            sf.write(os.path.join(para.save_path, f"samples/clean_e{epoch+1}_b{batch_idx}.wav"), clean_np, para.fs)
                        except Exception as e:
                            print("[WARN] could not write sample audio:", e)

            # epoch summary
            avg_d_loss = epoch_d_loss / max(1, steps)
            avg_l1 = epoch_l1 / max(1, steps)
            avg_d_real = d_real_sum / max(1, steps)
            avg_d_fake = d_fake_sum / max(1, steps)
            avg_g_loss_on_updates = epoch_g_loss_updates / max(1, epoch_g_updates) if epoch_g_updates > 0 else float('nan')
            avg_g_cond_on_updates = epoch_g_cond_updates / max(1, epoch_g_updates) if epoch_g_updates > 0 else float('nan')

            avg_d_history.append(avg_d_loss)
            avg_g_history.append(avg_g_loss_on_updates)
            avg_gcond_history.append(avg_g_cond_on_updates)
            avg_l1_history.append(avg_l1)
            avg_dreal_history.append(avg_d_real)
            avg_dfake_history.append(avg_d_fake)

            print(f"\nEpoch {epoch + 1} 统计:")
            print(f"平均 D_loss: {avg_d_loss:.4f}")
            print(f"平均 G_loss (仅更新时): {avg_g_loss_on_updates:.4f} (基于 {epoch_g_updates} 次更新)")
            print(f"平均 G_cond (仅更新时): {avg_g_cond_on_updates:.4f}")
            print(f"平均 实际L1: {avg_l1:.6f}")
            print(f"平均 D_real: {avg_d_real:.3f} | 平均 D_fake: {avg_d_fake:.3f}")

            # save only every SAVE_EVERY epochs
            if ((epoch + 1) % SAVE_EVERY) == 0:
                try:
                    os.makedirs(para.save_path, exist_ok=True)
                    # save model state_dicts (backward compatible)
                    gp_name = os.path.join(para.save_path, f"G_epoch{epoch + 1}_cond{avg_g_cond_on_updates:.4f}.pkl")
                    dp_name = os.path.join(para.save_path, f"D_epoch{epoch + 1}_loss{avg_d_loss:.4f}.pkl")
                    torch.save(generator.state_dict(), gp_name)
                    torch.save(discriminator.state_dict(), dp_name)
                    # save unified checkpoint with optimizer state
                    save_unified_checkpoint(para.save_path, epoch + 1, generator, discriminator, g_optimizer, d_optimizer)
                    print(f"[SAVE] Saved epoch {epoch + 1} checkpoints.")
                except Exception as e:
                    print("[WARN] failed to save epoch checkpoint:", e)

    except Exception as e:
        print("[FATAL] Exception during training:", e)
        traceback.print_exc()
        try:
            save_debug_checkpoint(para.save_path, epoch, generator, discriminator, g_optimizer, d_optimizer, suffix="exception")
        except Exception as e2:
            print("[WARN] failed to save debug checkpoint in exception handler:", e2)
        raise

    print("[INFO] Training finished normally.")
