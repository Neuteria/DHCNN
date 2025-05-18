import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json


class LayoutViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("布局预览器 v1.0")


        try:
            self.config = self.load_config("config.json")
            self.forbidden_areas = [
                {
                    "x": 0,
                    "y": 14,
                    "width": 8,
                    "height": 8,
                    "color": "#ff9999"
                },
                {
                    "x": 41,
                    "y": 12,
                    "width": 6,
                    "height": 6,
                    "color": "#ff9999"
                }
            ]
            self.root.geometry(f"{self.config["canvas_size"][1]*20+260}x{self.config["canvas_size"][0]*20}")
            self.setup_ui()
            self.create_menu()
        except Exception as e:
            self.show_error("初始化失败", str(e))
            sys.exit(1)

    def load_config(self, path):
        """加载配置文件并验证结构"""
        try:
            with open(path) as f:
                config = json.load(f)

            if "furniture_specs" not in config:
                raise ValueError("配置文件中缺少furniture_specs字段")

            return config
        except Exception as e:
            self.show_error("配置错误", str(e))
            raise

    def setup_ui(self):
        """初始化界面组件"""
        # 主画布区域
        self.canvas = tk.Canvas(self.root, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 信息面板
        info_frame = ttk.Frame(self.root, padding=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_label = ttk.Label(info_frame, text="当前文件: 无")
        self.file_label.pack(pady=5)

        ttk.Separator(info_frame).pack(fill=tk.X, pady=10)

        self.stats_label = ttk.Label(info_frame, text="家具统计:")
        self.stats_label.pack(anchor=tk.W)

    def create_menu(self):
        """创建菜单系统"""
        menubar = tk.Menu(self.root)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开布局文件", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)

        menubar.add_cascade(label="文件", menu=file_menu)
        self.root.config(menu=menubar)

    def open_file(self):
        """打开并解析布局文件"""
        file_path = filedialog.askopenfilename(
            title="选择布局文件",
            filetypes=[("JSON文件", "*.json")]
        )

        if not file_path:
            return

        try:
            with open(file_path) as f:
                layout_data = json.load(f)

            # 验证文件结构
            required_fields = ["canvas_size", "furniture"]
            if not all(field in layout_data for field in required_fields):
                raise ValueError("无效的布局文件格式")

            self.render_layout(layout_data, file_path)
            self.update_stats(layout_data)

        except Exception as e:
            self.show_error("加载错误", f"无法加载布局文件:\n{str(e)}")

    def render_layout(self, data, file_path):
        """在画布上渲染布局"""
        # 清除现有内容
        self.canvas.delete("all")
        self.file_label.config(text=f"当前文件: ...{file_path.split('DHCNN')[-1]}")

        # 设置画布尺寸
        canvas_width = data["canvas_size"][1] * 20
        canvas_height = data["canvas_size"][0] * 20
        self.canvas.config(width=canvas_width, height=canvas_height)

        # 绘制禁止区域
        for area in self.forbidden_areas:
            x1 = area["x"] * 20
            y1 = area["y"] * 20
            x2 = x1 + area["width"] * 20
            y2 = y1 + area["height"] * 20

            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=area["color"],
                outline="#ff6666",
                stipple="gray50"
            )
            self.canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2,
                text="禁止放置",
                fill="#ff4444",
                font=("宋体", 10, "bold"),
                angle=45
            )

        # 绘制家具
        furniture_stats = {}
        for item in data["furniture"]:
            # 获取家具规格
            spec = self.config["furniture_specs"].get(item["type"])
            if not spec:
                continue

            # 计算颜色
            red_value = int(spec["color"] * 255)
            color = f"#{red_value:02x}0000"

            # 计算位置和尺寸
            x = item["x"] * 20
            y = item["y"] * 20
            width = item["width"] * 20
            height = item["height"] * 20

            # 绘制家具
            self.canvas.create_rectangle(
                x, y, x + width, y + height,
                fill=color,
                outline="black",
                width=2
            )

            # 更新统计
            furniture_stats[item["type"]] = furniture_stats.get(item["type"], 0) + 1

    def update_stats(self, data):
        """更新统计信息"""
        stats_text = "家具统计:\n"
        counts = {}
        for item in data["furniture"]:
            counts[item["type"]] = counts.get(item["type"], 0) + 1

        for name, count in counts.items():
            stats_text += f"• {name}: {count}\n"

        self.stats_label.config(text=stats_text)

    def show_error(self, title, message):
        """显示错误对话框"""
        messagebox.showerror(title, message)


if __name__ == "__main__":
    LayoutViewer().root.mainloop()
