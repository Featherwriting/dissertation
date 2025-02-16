import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

# 配置参数
CONFIG = {
    "vocab": list("abcdefghijklmnopqrstuvwxyz%*+-/()#1234567890\n "),  # 自定义词汇表
    "max_len": 64,
    "batch_size": 512,
    "lr": 1e-4,
    "weight_decay": 0.5,
    "epochs": 30000,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# 数据预处理工具
class CodeDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path) as f:
            self.data = json.load(f)
        
        # 构建词汇表
        self.char2idx = {c:i for i,c in enumerate(CONFIG["vocab"])}
        self.idx2char = {i:c for i,c in enumerate(CONFIG["vocab"])}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # 将函数字符串转换为token索引
        tokens = [self.char2idx.get(c, 0) for c in sample["func"]][:CONFIG["max_len"]]
        padding = [0] * (CONFIG["max_len"] - len(tokens))
        tokens = torch.LongTensor(tokens + padding)
        outputs = torch.FloatTensor(sample["output"])
        return tokens, outputs

# 模型定义
class CodeGrokker(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(CONFIG["vocab"]), 256)
        self.pos_embed = nn.Embedding(CONFIG["max_len"], 256)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.3
            ),
            num_layers=6
        )
        self.head = nn.Sequential(
            nn.Linear(256, 1024),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1)
        )
    
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均池化
        return self.head(x)

# 训练循环
def train():
    # 初始化
    model = CodeGrokker().to(CONFIG["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion = nn.MSELoss()
    
    # 加载数据
    train_set = CodeDataset("train.json")
    test_set = CodeDataset("test.json")
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"])
    
    # 训练监控
    best_test_loss = float('inf')
    loss_history = []  # 存储损失历史

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        
        # 训练阶段
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=True):
            x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
            pred = model(x)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()
        
        # 计算平均训练损失
        avg_train = total_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
                pred = model(x)
                test_loss += criterion(pred, y).item()
        
        avg_test = test_loss / len(test_loader)
        
        # 存储损失
        loss_history.append({"epoch": epoch + 1, "train_loss": avg_train, "test_loss": avg_test})
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f}, Test Loss={avg_test:.4f}")

        scheduler.step()
    
    # 训练完成后保存损失历史
    with open("loss_log.json", "w") as f:
        json.dump(loss_history, f, indent=4)
    print("Loss history saved to loss_log.json")

if __name__ == "__main__":
    train()
