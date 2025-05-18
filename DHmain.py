import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder

import json
import os
from pathlib import Path
import re
import glob
import matplotlib.pyplot as plt

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config():
    """加载并验证配置文件"""
    try:
        # 获取当前文件所在目录
        config_path = Path(__file__).parent / "config.json"

        with open(config_path, 'r') as f:
            raw_config = json.load(f)

        # 转换数据结构
        config = {
            "canvas_size": tuple(raw_config["canvas_size"]),
            "forbidden_areas": raw_config["forbidden_areas"],
            "furniture_specs": {
                k: {
                    "size": tuple(v["size"]),
                    "count": int(v["count"]),
                    "color": float(v["color"])
                } for k, v in raw_config["furniture_specs"].items()
            },
            "train": {
                "batch_size": int(raw_config["train"]["batch_size"]),
                "epochs": int(raw_config["train"]["epochs"]),
                "lr": float(raw_config["train"]["lr"]),
                "heatmap_sigma": float(raw_config["train"]["heatmap_sigma"])
            }
        }

        # 验证必要字段
        required_keys = {"canvas_size", "furniture_specs", "train"}
        if not required_keys.issubset(config.keys()):
            missing = required_keys - config.keys()
            raise KeyError(missing)

        return config

    except FileNotFoundError:
        raise RuntimeError("配置文件 config.json 未找到")
    except json.JSONDecodeError:
        raise RuntimeError("配置文件格式错误")
    except KeyError as e:
        raise RuntimeError(f"配置缺少必要字段: {e}")
    except ValueError as e:
        raise RuntimeError(f"配置值类型错误: {e}")

#配置初始化
config = load_config()

# 数据加载器（带碰撞检测）
class FloorPlanGenerator:
    def __init__(self, config):
        self.le = LabelEncoder().fit(list(config['furniture_specs'].keys()))
        self.user_layouts = self.load_user_layouts()

    def load_user_layouts(self):
        """加载用户创建的布局"""
        layout_dir = Path(__file__).parent / "user_layouts"
        layouts = []

        for json_file in layout_dir.glob("*.json"):
            with open(json_file) as f:
                layout = json.load(f)

                # 验证布局格式
                if self.validate_layout(layout):
                    layouts.append(layout)
                else:
                    print(f"跳过无效布局文件: {json_file.name}")

        return layouts

    def validate_layout(self, layout):
        """验证布局文件有效性"""
        try:
            # 检查画布尺寸
            if tuple(layout['canvas_size']) != config['canvas_size']:
                return False

            # 检查家具配置
            counter = {name: 0 for name in config['furniture_specs']}
            for item in layout['furniture']:
                spec = config['furniture_specs'][item['type']]
                counter[item['type']] += 1

                # 检查尺寸是否匹配
                preset_size = spec['size']
                actual_size = (item['width'], item['height'])
                if not (actual_size == preset_size or
                        actual_size == preset_size[::-1]):
                    return False

            # 检查数量限制
            for name, count in counter.items():
                if count != config['furniture_specs'][name]['count']:
                    return False

            return True
        except:
            return False

    def generate_sample(self):
        """从用户布局生成样本"""
        layout = np.random.choice(self.user_layouts)

        num_classes = len(self.le.classes_)
        H, W = config['canvas_size']

        heatmap = np.zeros((2, H, W), dtype=np.float32)
        heatmap[0] = 1.0  # 房间区域

        grid = lambda a: int((a-a%4)/4)
        grid_H, grid_W = grid(H), grid(W)  # 模型输出的网格尺寸
        num_grids = grid_H * grid_W

        targets = torch.full((num_grids, 3 + num_classes), -1.0)  # 初始化为-1
        for area in enumerate(config['forbidden_areas']):
            x1 = area[1][0][0]
            y1 = area[1][0][1]
            x2 = area[1][1][0]
            y2 = area[1][1][1]

            # 标记热力图
            heatmap[0, y1:y2, x1:x2] = 0

        for item in layout['furniture']:
            x = item['x']
            y = item['y']
            w = item['width']
            h = item['height']
            orientation = 1 if item['rotated'] else 0

            # 标记热力图
            spec = config['furniture_specs'][item['type']]
            heatmap[1, y:y + h, x:x + w] = spec['color']

            x_center = x + w / 2
            y_center = y + h / 2

            grid_x = int(x_center / W * grid_W)
            grid_y = int(y_center / H * grid_H)

            grid_idx = (grid_y - 1) * grid_W + grid_x

            # 确保索引在范围内
            grid_idx = min(grid_idx, num_grids - 1)

            cls_idx = self.le.transform([item['type']])[0]
            # 生成One-Hot向量
            cls_onehot = torch.zeros(num_classes)
            cls_onehot[cls_idx] = 1.0

            # # 填充目标数据
            targets[grid_idx] = torch.cat([
                torch.tensor([grid_x, grid_y, orientation]),
                cls_onehot  # 拼接One-Hot向量
            ])

        return torch.FloatTensor(heatmap), targets


# 改进的CNN模型
class FurnitureLayoutCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 特征提取主干
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.6)
        )

        # 多任务预测头
        self.loc_head = nn.Sequential(
            nn.Conv2d(64, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 1),  # 预测x,y
        )

        self.orientation_head = nn.Sequential(
            nn.Conv2d(64, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),  # 预测旋转
        )

        # 修改后的分类头（输出单个通道）
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, num_classes, 1),  # 输出通道数=类别数
        )


    def forward(self, x):
        features = self.backbone(x)

        # 生成多任务预测
        locations = self.loc_head(features)
        orientations = self.orientation_head(features)
        class_probs = self.cls_head(features)

        # 需要将cls_probs从[batch, C, H, W]转为[batch, H*W, C+5]格式
        batch, C, H, W = class_probs.shape
        # 合并所有输出（需调整维度顺序）
        combined = torch.cat([
            locations.permute(0, 2, 3, 1).reshape(batch, -1, 2),  # 坐标 [batch, H*W, 2]
            orientations.permute(0, 2, 3, 1).reshape(batch, -1, 1),  # 旋转 [batch, H*W, 1]
            class_probs.permute(0, 2, 3, 1).reshape(batch, -1, C)  # 类别概率 [batch, H*W, C]
        ], dim=-1)

        return combined  # 最终形状: [batch, H*W, 5+C]


# 自定义损失函数
class FurnitureLoss(nn.Module):
    def __init__(self, specs):
        super().__init__()
        # 注册预设尺寸
        self.register_buffer('preset_sizes',
                             torch.tensor([v['size'] for v in specs.values()], dtype=torch.float32))

    def forward(self, preds, targets):
        valid_mask = (targets[..., 0] != -1)  # [batch, H*W]

        # 过滤无效目标
        preds_masked = preds[valid_mask]
        targets_masked = targets[valid_mask]

        # 分解预测结果
        pred_loc = preds_masked[..., :2]
        pred_rot = preds_masked[..., 2]
        pred_cls = preds_masked[..., 3:]

        # 分解目标
        true_loc = targets_masked[..., :2]
        true_rot = targets_masked[..., 2]
        true_cls = targets_masked[..., 3:].argmax(dim=-1)

        # 位置损失
        loc_loss = F.mse_loss(pred_loc, true_loc)
        # 旋转分类损失
        rot_loss = F.binary_cross_entropy_with_logits(pred_rot, true_rot)
        # 类别损失
        cls_loss = F.cross_entropy(
            pred_cls,  # 需要调整为[Batch, C, H*W]
            true_cls
        )
        total_loss = sum([loc_loss, rot_loss, cls_loss])

        return loc_loss * 0.1, rot_loss, cls_loss, total_loss


# 训练流程
def train():
    # 初始化
    generator = FloorPlanGenerator(config)
    model = FurnitureLayoutCNN(len(config['furniture_specs'])).to(device)
    criterion = FurnitureLoss(config['furniture_specs'])
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])

    train_losses = []
    loc_losses = []
    rot_losses = []
    cls_losses = []
    epoches = []

    # 训练循环
    for epoch in range(config['train']['epochs']):
        # 生成训练批次
        batch_heatmaps = []
        batch_targets = []
        for _ in range(config['train']['batch_size']):
            heatmap, targets = generator.generate_sample()
            batch_heatmaps.append(heatmap)
            batch_targets.append(targets)

        # 转换为张量
        inputs = torch.stack(batch_heatmaps).to(device)
        targets = torch.stack(batch_targets).to(device)

        # 前向传播
        outputs = model(inputs)

        # mask = targets.sum(dim=-1) != 0
        # outputs_masked = outputs[mask]
        # targets_masked = targets[mask]

        # 计算损失（仅计算有效目标）
        loc_loss, rot_loss, cls_loss, loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练信息
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{config["train"]["epochs"]}], Loss: {loss.item():.4f}')
            train_losses.append(loss.item())
            loc_losses.append(loc_loss.item())
            rot_losses.append(rot_loss.item())
            cls_losses.append(cls_loss.item())
            epoches.append(epoch)

    plt.figure(figsize=(8, 4))
    plt.plot(epoches, train_losses, label='Total Loss', linewidth=2)
    plt.plot(epoches, loc_losses, label='Location Loss', linewidth=2)
    plt.plot(epoches, rot_losses, label='Rotation Loss', linewidth=2)
    plt.plot(epoches, cls_losses, label='Type Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

    return model


# 后处理与验证
def validate_layout(preds, config):
    """将模型输出转换为实际坐标并验证约束，返回带类型信息的框"""
    H, W = config['canvas_size']
    grid = lambda a: int((a - a % 4) / 4)
    grid_H, grid_W = grid(H), grid(W)
    valid_boxes = []

    for idx in range(grid_H * grid_W):
        cls_logits = preds[idx][3:]  # 类别概率向量
        cls_probs = F.softmax(cls_logits, dim=-1)
        cls_idx = torch.argmax(cls_probs)  # 选择最高概率的类别
        cls_name = list(config['furniture_specs'].keys())[cls_idx]

        # 检查概率是否超过阈值（如0.5）
        if cls_probs[cls_idx] < 0.9:
            continue

        x_center, y_center, ro = preds[idx][:3]

        # 应用预设尺寸
        preset_w, preset_h = config['furniture_specs'][cls_name]['size']
        if ro > 0.5:
            width, height = preset_h, preset_w
        else:
            width, height = preset_w, preset_h

        # 转换为实际坐标
        x = int(x_center / grid_W * W - width / 2)
        y = int(y_center / grid_H * H - height / 2)

        # 检查边界
        if x < 0 or y < 0 or x + width > W or y + height > H:
            continue

        # 检查碰撞
        new_box = (x, y, width, height, cls_name, ro)
        if idx == 0 :
            valid_boxes.append(new_box)
            continue

        if not _check_collision(new_box[0:5], valid_boxes):
            valid_boxes.append(new_box)

    return valid_boxes


def _check_collision(new_box, existing_boxes):
    """修正后的碰撞检测函数"""
    # 解包新家具信息 [x, y, w, h, cls_name, ro]
    x1, y1, w1, h1, new_cls = new_box
    # 转换坐标为（x_min, y_min, x_max, y_max）
    ax1, ay1 = x1, y1
    ax2, ay2 = x1 + w1, y1 + h1

    for existing_box in existing_boxes:
        # 解包已有家具信息
        x2, y2, w2, h2, exist_cls, _ = existing_box
        bx1, by1 = x2, y2
        bx2, by2 = x2 + w2, y2 + h2

        # 柜子特殊处理：不与任何物体碰撞（包括其他柜子）
        # if ("cabinet" in [new_cls, exist_cls]) and ("bed" in [new_cls, exist_cls]):
        #         continue

        # 分离轴定理核心判断
        overlap_x = (ax1 - 2 <= bx2) and (ax2 >= (bx1 - 2))
        overlap_y = (ay1 - 2 <= by2) and (ay2 >= (by1 - 2))

        # 仅在两个轴都有重叠时返回碰撞
        if overlap_x and overlap_y:
            return True

    return False

# 修改后的save_validated_layout函数
def save_validated_layout(preds, config, save_dir="generated_layouts"):
    """保存验证通过的布局为JSON文件"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 验证布局并获取有效家具框（现在包含类型信息）
    valid_boxes = validate_layout(preds, config)
    print(len(valid_boxes))
    # 创建类型计数字典
    type_counter = {}
    # 初始化配置计数器
    config_counts = {k: v['count'] for k, v in config['furniture_specs'].items()}

    # 统计实际生成的家具类型
    for box in valid_boxes:
        furniture_type = box[4]
        type_counter[furniture_type] = type_counter.get(furniture_type, 0) + 1

    # 检查所有配置类型数量匹配
    for ftype, expected_count in config_counts.items():
        actual_count = type_counter.get(ftype, 0)
        if ftype == 'bed':
            # print(f'床数量：{actual_count}')
            if (actual_count < expected_count - 1) or (actual_count > expected_count):
                return 0
        else:
            if actual_count != expected_count:
        #         print(f"类型 {ftype} 数量不匹配: 需要 {expected_count}, 生成 {actual_count}")
        # if actual_count != expected_count:
                return 0

    # total_count = sum(v['count'] for v in config['furniture_specs'].values())
    print(len(valid_boxes))
    # if len(valid_boxes) != total_count:
    #     return 0

    # 转换为JSON格式
    layout_data = {
        "canvas_size": list(config['canvas_size']),
        "furniture": []
    }

    # 为每个有效框创建家具条目
    for x, y, w, h, cls_name, ro in valid_boxes:
        layout_data["furniture"].append({
            "type": cls_name,
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "rotated": int(ro >= 0.5)  # 将旋转概率转换为0/1
        })

    # 生成文件名
    filename = get_next_filename()
    filepath = os.path.join(save_dir, filename)

    # 保存为JSON文件
    with open(filepath, 'w') as f:
        json.dump(layout_data, f, indent=2)

    print(f"布局已保存到: {filepath}")
    return filepath

def get_next_filename():
    """查找第一个可用的编号文件名"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "generated_layouts")
    existing = []
    pattern = re.compile(r'layout_(\d{3})\.json$')

    os.makedirs(save_dir, exist_ok=True)

    # 修复路径：使用绝对路径匹配
    search_pattern = os.path.join(save_dir, "layout_*.json")

    # 扫描所有符合格式的文件
    for fname in glob.glob(search_pattern):
        match = pattern.search(fname)
        if match:
            existing.append(int(match.group(1)))

    # 如果没有现存文件
    if not existing:
        return "layout_001.json"

    # 排序并查找空缺
    existing.sort()
    full_range = set(range(1, existing[-1] + 2))  # +2确保包含末尾+1

    # 找到第一个缺失的编号
    for num in full_range:
        if num not in existing:
            return f"layout_{num:03d}.json"

    # 如果所有编号连续，返回下一个
    return f"layout_{existing[-1] + 1:03d}.json"



# 在训练后使用示例
if __name__ == "__main__":
    model = train()
    # print("模型参数量:", sum(p.numel() for p in model.parameters()))

    # 生成并保存布局
    with torch.no_grad():
        generator = FloorPlanGenerator(config)
        for i in range(5000):
            if (i % 1000 == 0) and (i != 0):
                print(f"已执行{i}次")
            sample_heatmap, _ = generator.generate_sample()
            preds = model(sample_heatmap.unsqueeze(0).to(device))
            save_validated_layout(preds[0].cpu(), config)

