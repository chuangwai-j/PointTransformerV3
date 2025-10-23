import torch, yaml
from pointcept.models import build_model

cfg_path = "configs/wind_shear/pointtransformer_v3.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 建 5 类空壳
cfg['model']['num_classes'] = 5
model = build_model(cfg['model']).to(device)

# 2. 加载两段权重（Pointcept 格式）
sd1 = torch.load('checkpoints/checkpoints_3/best_stage1_model.pth', map_location=device)['model_state_dict']
sd2 = torch.load('checkpoints/checkpoints_3/best_stage2_model.pth', map_location=device)['model_state_dict']

# 3. 组装新 state_dict
new_sd = model.state_dict()

# 3-1 backbone：复用 stage1 的（除 head 外全部一致）
for k, v in sd1.items():
    if k not in ('head.3.weight', 'head.3.bias'):
        new_sd[k] = v

# 3-2 拼接 head
with torch.no_grad():
    new_sd['head.3.weight'][0:2] = sd1['head.3.weight'][0:2]   # 0,1
    new_sd['head.3.bias'][0:2]   = sd1['head.3.bias'][0:2]
    new_sd['head.3.weight'][2:5] = sd2['head.3.weight'][2:5]   # 2,3,4
    new_sd['head.3.bias'][2:5]   = sd2['head.3.bias'][2:5]

# 4. 写回并保存
model.load_state_dict(new_sd)
torch.save(model.state_dict(), 'checkpoints/checkpoints_3/best_merged_model.pth')
print('✅ 已保存 merged 5 类模型 -> checkpoints/checkpoints_3/best_merged_model.pth')
