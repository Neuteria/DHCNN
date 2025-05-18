import sys
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import json
from pathlib import Path
import os
import glob
import re


# 修改加载配置部分，添加详细错误提示
def load_config(path):
    try:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")

        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        # 显示图形化错误提示
        tk.Tk().withdraw()
        messagebox.showerror("配置错误", f"加载配置文件失败:\n{str(e)}")
        sys.exit(1)

class FurnitureEditor:
    def __init__(self, config_path):
        try:
            self.config = load_config(config_path)
            self.canvas_size = self.config['canvas_size']
            self.setup_states()
            self.setup_ui()


        except Exception as e:
            tk.Tk().withdraw()
            messagebox.showerror("初始化错误", f"程序初始化失败:\n{str(e)}")
            sys.exit(1)


    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("家具布局编辑器 v2.0")

        # 主画布区域
        self.canvas = tk.Canvas(self.root,
                                width=self.canvas_size[1] * 20,  # 放大显示
                                height=self.canvas_size[0] * 20,
                                bg='white')
        self.draw_forbidden_areas()
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 控制面板
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y)

        # 家具选择
        self.selected_furniture = tk.StringVar()
        ttk.Label(control_frame, text="选择家具:").grid(row=0, column=0, sticky='w')
        for i, (name, spec) in enumerate(self.config['furniture_specs'].items(), 1):
            rb = ttk.Radiobutton(control_frame,
                                 text=f"{name} ({spec['size']})",
                                 variable=self.selected_furniture,
                                 value=name)
            rb.grid(row=i, column=0, sticky='w')

        # 方向切换
        ttk.Button(control_frame,
                   text="切换方向 (R)",
                   command=self.toggle_orientation).grid(row=10, column=0, pady=5)

        # 对齐模式
        self.align_mode = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame,
                        text="网格对齐",
                        variable=self.align_mode).grid(row=11, column=0)

        # 计数显示
        self.count_labels = {}
        for i, (name, spec) in enumerate(self.config['furniture_specs'].items(), 1):
            label = ttk.Label(control_frame, text=f"已放置: 0/{spec['count']}")
            label.grid(row=i, column=1, padx=10)
            self.count_labels[name] = label

        # 操作按钮
        ttk.Button(control_frame,
                   text="保存布局 (Ctrl+S)",
                   command=self.save_layout).grid(row=20, column=0, pady=20)
        ttk.Button(control_frame,
                   text="撤销 (Ctrl+Z)",
                   command=self.undo).grid(row=21, column=0)

        # 坐标显示
        self.coord_label = ttk.Label(control_frame, text="坐标: (0, 0)")
        self.coord_label.grid(row=22, column=0)

        # 事件绑定
        self.canvas.bind("<Motion>", self.show_preview)
        self.canvas.bind("<Button-1>", self.place_furniture)
        self.canvas.bind("<Button-3>", self.delete_furniture)
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-s>", lambda e: self.save_layout())
        self.root.bind("<r>", self.toggle_orientation)
        self.root.mainloop()

    def draw_forbidden_areas(self):
        """绘制不可放置区域"""
        for area in enumerate(self.config['forbidden_areas']):
            x1 = area[1][0][0] * 20
            y1 = area[1][0][1] * 20
            x2 = area[1][1][0] * 20
            y2 = area[1][1][1] * 20

            # 绘制半透明效果
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill="#ff9999",
                outline="#ff6666",  # 边框颜色
                stipple="gray50",  # 半透明纹理
                tags="forbidden_area"
            )

            # 添加警示文字
            self.canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2,
                text="禁止放置",
                fill="#ff4444",
                font=("宋体", 10, "bold"),
                angle=45
            )


    def setup_states(self):
        self.placed_items = {name: [] for name in self.config['furniture_specs']}
        self.current_orientation = 0  # 0-正常方向，1-旋转
        self.preview_rect = None
        self.history = []

    # 新增功能实现 --------------------------------------------------------
    def toggle_orientation(self, event=None):
        """切换家具方向（兼容事件参数）"""
        self.current_orientation = 1 - self.current_orientation
        # 如果从键盘事件触发，自动获取鼠标当前位置
        if not event:
            # 获取屏幕绝对坐标
            abs_x, abs_y = self.root.winfo_pointerxy()
            # 转换为画布相对坐标
            canvas_x = abs_x - self.canvas.winfo_rootx()
            canvas_y = abs_y - self.canvas.winfo_rooty()
            event = type('FakeEvent', (), {'x': canvas_x, 'y': canvas_y})()
        self.show_preview(event)

    def show_preview(self, event=None):
        """显示放置预览"""
        try:
            # 清除旧预览
            if self.preview_rect and self.canvas.winfo_exists():
                self.canvas.delete(self.preview_rect)
                self.preview_rect = None

            # 有效性检查
            if not self.selected_furniture.get():
                return

            # 自动获取鼠标位置（当无事件时）
            if not event:
                abs_x, abs_y = self.root.winfo_pointerxy()
                canvas_x = abs_x - self.canvas.winfo_rootx()
                canvas_y = abs_y - self.canvas.winfo_rooty()
                event = type('', (), {'x': max(0, min(canvas_x, self.canvas.winfo_width())),
                                      'y': max(0, min(canvas_y, self.canvas.winfo_height()))})()

            # 坐标边界保护
            event.x = max(0, min(event.x, self.canvas.winfo_width()))
            event.y = max(0, min(event.y, self.canvas.winfo_height()))

            # 获取当前家具规格
            name = self.selected_furniture.get()

            # 计算尺寸
            spec = self.config['furniture_specs'][name]
            w, h = spec['size']
            if self.current_orientation == 1:
                w, h = h, w

            # 计算坐标
            grid_x, grid_y = self.get_aligned_position(event.x, event.y)
            x1 = grid_x * 20
            y1 = grid_y * 20
            x2 = x1 + w * 20
            y2 = y1 + h * 20

            # 绘制半透明预览
            self.preview_rect = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill="#%02x0000" % int(spec['color'] * 255),
                stipple="gray50",
                outline="gray"
            )

            # 更新坐标显示
            if event:
                self.coord_label.config(text=f"坐标: ({grid_x}, {grid_y})")

        except Exception as e:
            print(f"预览异常: {str(e)}")

    def get_aligned_position(self, x, y):
        """基于旧数据的对齐逻辑"""
        # 转换为网格坐标
        grid_x = x // 20
        grid_y = y // 20

        if self.align_mode.get():
            return grid_x, grid_y

        # 精确吸附逻辑（毫米级精度）
        min_dist = 0.5  # 吸附阈值（0.5个网格单位）
        raw_x = x / 20
        raw_y = y / 20

        # 搜索所有已放置家具
        for category in self.placed_items:
            for item in self.placed_items[category]:
                # 获取家具四边坐标
                left = item['x']
                right = left + item['width']
                top = item['y']
                bottom = top + item['height']

                # X轴吸附检测
                for edge in [left, right]:
                    if abs(raw_x - edge) < min_dist:
                        grid_x = edge
                        min_dist = abs(raw_x - edge)

                # Y轴吸附检测
                for edge in [top, bottom]:
                    if abs(raw_y - edge) < min_dist:
                        grid_y = edge
                        min_dist = abs(raw_y - edge)

        return int(grid_x) if isinstance(grid_x, float) and grid_x.is_integer() else round(grid_x, 2), \
            int(grid_y) if isinstance(grid_y, float) and grid_y.is_integer() else round(grid_y, 2)

    def place_furniture(self, event):
        """保持旧数据结构并修复碰撞检测"""
        name = self.selected_furniture.get()
        if not name:
            return

        spec = self.config['furniture_specs'][name]
        if len(self.placed_items[name]) >= spec['count']:
            messagebox.showwarning("提示", f"已达到{name}的最大数量限制")
            return

        # 计算位置和尺寸
        grid_x, grid_y = self.get_aligned_position(event.x, event.y)
        base_w, base_h = spec['size']
        if self.current_orientation == 1:
            w, h = base_h, base_w
        else:
            w, h = base_w, base_h

        # 创建旧格式数据
        new_item = {
            "type": name,  # 新增type字段
            "x": grid_x,
            "y": grid_y,
            "width": w,
            "height": h,
            "rotated": self.current_orientation,
            "canvas_id": self.canvas.create_rectangle(  # 直接创建图形
                grid_x * 20, grid_y * 20,
                (grid_x + w) * 20, (grid_y + h) * 20,
                fill="#%02x0000" % int(spec['color'] * 255),
                outline='black'
            )
        }

        # 碰撞检测
        if self.check_collision(new_item):
            self.canvas.delete(new_item["canvas_id"])
            messagebox.showwarning("碰撞警告", "该位置已有家具！")
            return

        # 存储数据
        self.placed_items[name].append(new_item)
        self.history.append(('add', name, new_item))
        self.update_count(name)

    def is_overlap(self, new_item):
        """检查新家具是否与已有家具重叠"""
        new_left = new_item['position'][0]
        new_top = new_item['position'][1]
        new_right = new_left + new_item['size'][0]
        new_bottom = new_top + new_item['size'][1]

        for category in self.placed_items:
            for item in self.placed_items[category]:
                # 获取现有家具边界
                exist_left = item['position'][0]
                exist_top = item['position'][1]
                exist_right = exist_left + item['size'][0]
                exist_bottom = exist_top + item['size'][1]

                # 碰撞检测算法
                if not (new_right <= exist_left or
                        new_left >= exist_right or
                        new_bottom <= exist_top or
                        new_top >= exist_bottom):
                    return True
        return False

    def delete_furniture(self, event):
        """右键删除家具"""
        items = self.canvas.find_overlapping(event.x - 1, event.y - 1, event.x + 1, event.y + 1)
        if not items:
            return

        for item in items:
            # 查找对应的家具记录
            for name in self.placed_items:
                for idx, placed in enumerate(self.placed_items[name]):
                    if placed['id'] == item:
                        # 记录历史
                        self.history.append(('delete', name, placed))
                        # 删除画布对象
                        self.canvas.delete(item)
                        # 移除记录
                        del self.placed_items[name][idx]
                        self.update_count(name)
                        return

    def undo(self):
        """兼容旧数据结构的撤销功能"""
        if not self.history:
            return

        action = self.history.pop()
        if action[0] == 'add':
            _, name, data = action
            self.canvas.delete(data['canvas_id'])
            self.placed_items[name] = [item for item in self.placed_items[name]
                                       if item['canvas_id'] != data['canvas_id']]
            self.update_count(name)
        elif action[0] == 'delete':
            _, name, data = action
            # 恢复图形
            data['canvas_id'] = self.canvas.create_rectangle(
                data['x'] * 20, data['y'] * 20,
                (data['x'] + data['width']) * 20,
                (data['y'] + data['height']) * 20,
                fill="#%02x0000" % int(self.config['furniture_specs'][name]['color'] * 255),
                outline='black'
            )
            self.placed_items[name].append(data)
            self.update_count(name)

    def check_collision(self, new_item):
        """基于旧数据结构的碰撞检测"""
        new_x = new_item['x']
        new_y = new_item['y']
        new_w = new_item['width']
        new_h = new_item['height']

        for category in self.placed_items:
            for item in self.placed_items[category]:
                # 排除自己
                if item.get('canvas_id') == new_item.get('canvas_id'):
                    continue

                exist_x = item['x']
                exist_y = item['y']
                exist_w = item['width']
                exist_h = item['height']

                # AABB碰撞检测
                if not (new_x >= exist_x + exist_w or
                        new_x + new_w <= exist_x or
                        new_y >= exist_y + exist_h or
                        new_y + new_h <= exist_y):
                    return True
        return False

    def update_count(self, name):
        self.count_labels[name].config(
            text=f"已放置: {len(self.placed_items[name])}/{self.config['furniture_specs'][name]['count']}")

    def save_layout(self):
        """保存布局到文件"""
        try:
            """保存到user_layouts目录"""
            # 获取当前代码所在目录的路径
            base_dir = os.path.dirname(os.path.abspath(__file__))

            # 创建保存目录
            save_dir = os.path.join(base_dir, "user_layouts")
            os.makedirs(save_dir, exist_ok=True)  # 自动创建目录
            # 获取下一个可用文件名
            filename = self.get_next_filename()

            # 构建保存数据
            save_data = {
                "canvas_size": self.canvas_size,
                "furniture": self.get_current_layout()
            }

            # 写入文件
            save_path = os.path.join(save_dir, filename)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)

            messagebox.showinfo("保存成功", f"平面图已保存为：\n{filename}")

        except json.JSONDecodeError as je:
            messagebox.showerror("序列化失败", f"无法序列化数据：{str(je)}")
        except ValueError as ve:
            messagebox.showerror("数据校验失败", str(ve))
        except Exception as e:
            messagebox.showerror("保存失败", f"未处理的异常：{str(e)}")

    def get_next_filename(self):
        """查找第一个可用的编号文件名"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, "user_layouts")
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

    def get_current_layout(self):
        """生成符合要求的JSON结构"""
        layout = []
        for category, items in self.placed_items.items():
            for item in items:
                layout.append({
                    "type": category,
                    "x": item['x'],
                    "y": item['y'],
                    "width": item['width'],
                    "height": item['height'],
                    "rotated": item['rotated']
                })
        return layout


if __name__ == "__main__":
    editor = FurnitureEditor("config.json")
