# train.py
import torch
from dataset import SEGAN_Dataset  # 你的 dataset 文件名
from hparams import hparams
from model import Generator, Discriminator
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import traceback

def safe_tensor_stats(tensor):
    if tensor is None:
        return "None"
    try:
        return f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}"
    except:
        return str(tensor.shape)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    para = hparams()

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=para.lr_G, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=para.lr_D, betas=(0.5, 0.999))

    dataset = SEGAN_Dataset(para)
    # prepare a single reference batch and move to device (固定参考)
    ref_batch = dataset.ref_batch(para.ref_batch_size).to(device)

    dataloader = DataLoader(dataset, batch_size=para.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"\n数据信息:")
    print(f"总样本数: {len(dataset)}")
    print(f"每个epoch的批次: {len(dataloader)}")
    print(f"开始训练...\n")

    for epoch in range(35,para.n_epoch):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{para.n_epoch}")
        print(f"{'=' * 50}")

        epoch_g_loss_updates = 0.0
        epoch_g_cond_updates = 0.0
        epoch_g_updates = 0
        epoch_d_loss = 0.0
        epoch_l1 = 0.0
        d_real_sum = 0.0
        d_fake_sum = 0.0
        steps = 0

        try:
            for batch_idx, (batch_clean, batch_noisy) in enumerate(dataloader):
                steps += 1
                batch_clean = batch_clean.to(device)
                batch_noisy = batch_noisy.to(device)

                batch_z = torch.randn(batch_clean.size(0), para.size_z[0], para.size_z[1], device=device)

                # ========== 1. Train D ==========
                discriminator.zero_grad()

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

                # NaN/Inf check
                if torch.isnan(d_loss) or torch.isinf(d_loss):
                    print("Detected NaN/Inf in d_loss -- aborting. Stats:")
                    print("real_outputs:", safe_tensor_stats(real_outputs))
                    print("fake_outputs:", safe_tensor_stats(fake_outputs))
                    raise RuntimeError("d_loss NaN/Inf")

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                d_optimizer.step()

                # ========== 2. Train G ==========
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

                    # NaN/Inf check
                    if torch.isnan(g_loss) or torch.isinf(g_loss):
                        print("Detected NaN/Inf in g_loss -- aborting. Stats:")
                        print("outputs:", safe_tensor_stats(outputs))
                        print("generated:", safe_tensor_stats(generated))
                        raise RuntimeError("g_loss NaN/Inf")

                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    g_optimizer.step()

                    epoch_g_loss_updates += g_loss.item()
                    epoch_g_cond_updates += g_cond_loss.item()
                    epoch_g_updates += 1

                # accumulate metrics
                epoch_d_loss += d_loss.item()
                epoch_l1 += torch.mean(torch.abs(generated - batch_clean)).item()
                d_real_sum += real_outputs.mean().item()
                d_fake_sum += fake_outputs.mean().item()

                # periodic print (use generated from most recent forward)
                if batch_idx % 100 == 0:
                    cur_l1 = torch.mean(torch.abs(generated - batch_clean)).item()
                    print(f"Batch {batch_idx:4d}/{len(dataloader)} | "
                          f"D_loss: {d_loss.item():.4f} | "
                          f"G_loss: {g_loss.item() if do_update_g else float('nan'):.4f} | "
                          f"G_cond: {g_cond_loss.item() if do_update_g else float('nan'):.4f}")
                    print(f"          实际L1: {cur_l1:.4f} | "
                          f"D_real: {real_outputs.mean().item():.3f} | "
                          f"D_fake: {fake_outputs.mean().item():.3f}")

            # epoch summary
            avg_d_loss = epoch_d_loss / max(1, steps)
            avg_l1 = epoch_l1 / max(1, steps)
            avg_d_real = d_real_sum / max(1, steps)
            avg_d_fake = d_fake_sum / max(1, steps)
            avg_g_loss_on_updates = epoch_g_loss_updates / max(1, epoch_g_updates)
            avg_g_cond_on_updates = epoch_g_cond_updates / max(1, epoch_g_updates)

            print(f"\nEpoch {epoch + 1} 统计:")
            print(f"平均 D_loss: {avg_d_loss:.4f}")
            print(f"平均 G_loss (仅更新时): {avg_g_loss_on_updates:.4f} (基于 {epoch_g_updates} 次更新)")
            print(f"平均 G_cond (仅更新时): {avg_g_cond_on_updates:.4f}")
            print(f"平均 实际L1: {avg_l1:.6f}")
            print(f"平均 D_real: {avg_d_real:.3f} | 平均 D_fake: {avg_d_fake:.3f}")

            # save checkpoints periodically
            if (epoch + 1) % 10 == 0 or epoch == 0:
                os.makedirs(para.save_path, exist_ok=True)
                g_path = os.path.join(para.save_path, f"G_epoch{epoch + 1}_cond{avg_g_cond_on_updates:.4f}.pkl")
                d_path = os.path.join(para.save_path, f"D_epoch{epoch + 1}_loss{avg_d_loss:.4f}.pkl")
                torch.save(generator.state_dict(), g_path)
                torch.save(discriminator.state_dict(), d_path)
                print(f"模型已保存: {g_path}")

        except Exception as e:
            print("Exception during training loop:", e)
            traceback.print_exc()
            # optional: save last models for debugging
            try:
                os.makedirs(para.save_path, exist_ok=True)
                torch.save(generator.state_dict(), os.path.join(para.save_path, "G_error_state.pth"))
                torch.save(discriminator.state_dict(), os.path.join(para.save_path, "D_error_state.pth"))
                print("Saved model states after exception.")
            except Exception as e2:
                print("Failed to save models after exception:", e2)
            # re-raise or break
            raise
