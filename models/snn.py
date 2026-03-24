import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen

class SHD_SNN(nn.Module):
    def __init__(self, input_size=700, hidden_size=256, output_size=20, beta=0.9):
        super().__init__()

        # 全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # LIF神经元层
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # x shape: [batch, time_steps, 1, 700]
        x = x.squeeze(2)  # → [batch, time_steps, 700]

        batch_size, time_steps, _ = x.shape

        # 初始化膜电位
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # 记录输出层的spike和膜电位
        spk2_rec = []
        mem2_rec = []

        # 按时间步迭代
        for t in range(time_steps):
            cur1 = self.fc1(x[:, t, :])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        # stack成 [time_steps, batch, output_size]
        return torch.stack(spk2_rec), torch.stack(mem2_rec)