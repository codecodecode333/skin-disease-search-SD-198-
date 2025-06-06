import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from search_similar import run_similarity_search

class ImageSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 피부 질환 이미지 검색")
        self.query_path = None
        self.result_imgs = []

        self.setup_ui()

    def setup_ui(self):
        # 버튼 영역
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.grid(row=0, column=0, sticky="ew")

        ttk.Button(btn_frame, text="🔍 쿼리 이미지 선택", command=self.select_image).grid(row=0, column=0)

        # 이미지 표시 영역
        self.image_frame = ttk.Frame(self.root, padding=10)
        self.image_frame.grid(row=1, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.query_path = file_path
            self.display_results()

    def display_results(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        ttk.Label(self.image_frame, text="Query Image", font=("Arial", 12, "bold")).grid(row=0, column=0)
        query_img = self.load_image(self.query_path, (224, 224))
        ttk.Label(self.image_frame, image=query_img).grid(row=1, column=0, padx=10, pady=10)
        self.query_img = query_img  # 참조 유지

        # 검색 실행
        results = run_similarity_search(self.query_path, mode="topk")

        # 결과 표시
        result_frame = ttk.Frame(self.image_frame)
        result_frame.grid(row=1, column=1, padx=10)

        ttk.Label(self.image_frame, text="Top-5 유사 이미지", font=("Arial", 12, "bold")).grid(row=0, column=1)

        for i, (item, sim) in enumerate(results):
            img = self.load_image(item["path"], (150, 150))
            ttk.Label(result_frame, image=img).grid(row=i, column=0, padx=5, pady=5)
            ttk.Label(result_frame, text=f"{item['class']}\n유사도: {sim:.4f}",
                      font=("Arial", 10)).grid(row=i, column=1, sticky="w")
            self.result_imgs.append(img)  # 참조 유지

    def load_image(self, path, size=(150, 150)):
        img = Image.open(path).resize(size)
        return ImageTk.PhotoImage(img)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchGUI(root)
    root.mainloop()
