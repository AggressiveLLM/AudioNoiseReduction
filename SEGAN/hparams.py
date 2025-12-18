# hparams.py
import os


class hparams():
    def __init__(self):
        # ========== 数据路径 ========== #
        self.train_scp = "scp/train_segan.scp"

        # ========== 音频参数 ========== #
        self.fs = 16000
        self.win_len = 16384

        # ========== 训练参数 ========== #
        self.batch_size = 16
        self.ref_batch_size = 16

        # 学习率（可根据实验调整）
        self.lr_G = 2e-5
        self.lr_D = 1e-5

        # 训练轮次
        self.n_epoch = 100

        # G 更新频率：每多少个 batch 更新一次 G（推荐 1）
        self.g_update_freq = 1

        # ========== 模型参数 ========== #
        self.size_z = (1024, 8)

        # ========== 损失权重（修改为更平衡） ========== #
        self.w_g_loss1 = 0.5  # 对抗损失权重
        self.w_g_loss2 = 10   # 条件 L1 权重：从 100 改为 10，先试 10

        # ========== 保存路径 ========== #
        self.save_path = "checkpoints"
        os.makedirs(self.save_path, exist_ok=True)

        # 打印配置
        self.print_config()

    def print_config(self):
        print("=" * 60)
        print("SEGAN 训练配置")
        print("=" * 60)
        print(f"数据集: {self.train_scp}")
        print(f"批次大小: {self.batch_size} (干净+噪声: {self.batch_size * 2})")
        print(f"参考批次: {self.ref_batch_size}")
        print(f"学习率 - 生成器: {self.lr_G:.1e}, 判别器: {self.lr_D:.1e}")
        print(f"G 更新频率 (batch): {self.g_update_freq}")
        print(f"噪声维度: {self.size_z}")
        print(f"条件损失权重: {self.w_g_loss2}")
        print(f"保存路径: {self.save_path}")
        print(f"总训练轮次: {self.n_epoch}")
        print("=" * 60)
