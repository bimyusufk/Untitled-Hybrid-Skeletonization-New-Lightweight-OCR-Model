import os
import tkinter as tk
from tkinter import ttk
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =====================================================================
# CONFIGURATION PATHS
# =====================================================================
DATASET_DIR = "datasets"
STAGES = ["raw", "otsu-thresholding", "hole-filling", "skeletonize"]

class DatasetViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TA Preprocessing Viewer - Hybrid OCR")
        self.root.geometry("1200x650")
        
        # Ambil daftar semua subfolder Sample* yang ada di folder raw
        self.raw_base_dir = os.path.join(DATASET_DIR, "raw")
        if not os.path.exists(self.raw_base_dir):
            # Fallback jika folder mentahnya bernama lain atau langsung di bawah datasets
            self.raw_base_dir = DATASET_DIR
            
        self.samples = sorted([d for d in os.listdir(self.raw_base_dir) if d.lower().startswith("sample")])
        
        if not self.samples:
            print("Error: Tidak ditemukan folder 'Sample*' di dalam direktori data.")
            return

        # =====================================================================
        # GUI CONTROLS FRAME (ATAS)
        # =====================================================================
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Dropdown untuk Pilih Label/Folder Sample
        ttk.Label(control_frame, text="Pilih Folder Label:").pack(side=tk.LEFT, padx=5)
        self.sample_cb = ttk.Combobox(control_frame, values=self.samples, state="readonly", width=20)
        self.sample_cb.pack(side=tk.LEFT, padx=5)
        self.sample_cb.bind("<<ComboboxSelected>>", self.on_sample_selected)
        
        # Dropdown untuk Pilih File Gambar
        ttk.Label(control_frame, text="Pilih File Gambar:").pack(side=tk.LEFT, padx=15)
        self.image_cb = ttk.Combobox(control_frame, state="readonly", width=25)
        self.image_cb.pack(side=tk.LEFT, padx=5)
        self.image_cb.bind("<<ComboboxSelected>>", self.update_plots)

        # =====================================================================
        # PLOT CANVAS FRAME (BAWAH)
        # =====================================================================
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Inisialisasi Matplotlib Figure dengan 4 Axes sejajar
        self.fig, self.axes = plt.subplots(1, 4, figsize=(14, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Set default selection awal
        self.sample_cb.current(0)
        self.on_sample_selected(None)

    def on_sample_selected(self, event):
        """Dipanggil saat dropdown folder Sample diubah untuk mengupdate isi list gambar"""
        selected_sample = self.sample_cb.get()
        sample_path = os.path.join(self.raw_base_dir, selected_sample)
        
        # Ambil semua file gambar yang ada di folder sample terpilih
        if os.path.exists(sample_path):
            images = sorted([f for f in os.listdir(sample_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            self.image_cb['values'] = images
            if images:
                self.image_cb.current(0)
                self.update_plots(None)
            else:
                self.image_cb.set('')
                self.clear_plots("Folder Kosong")
        else:
            self.clear_plots("Folder Tidak Ditemukan")

    def update_plots(self, event):
        """Memuat dan menampilkan 4 gambar secara sejajar dari tiap tahap"""
        selected_sample = self.sample_cb.get()
        selected_image = self.image_cb.get()
        
        if not selected_image:
            return
            
        titles = ["1. Raw Image", "2. Otsu Threshold", "3. Hole Filling", "4. Skeletonize (1px)"]
        
        for i, stage in enumerate(STAGES):
            self.axes[i].clear()
            
            # Bangun path gambar untuk setiap tahapan spesifik
            # Menangani jika struktur foldermu adalah datasets/[stage]/Sample*
            img_path = os.path.join(DATASET_DIR, stage, selected_sample, selected_image)
            
            # Pengaman jika di skrip sebelumnya foldermu bernama 'otsu-thresholding' tapi dicari 'otsu-threshold'
            if not os.path.exists(img_path) and stage == "otsu-threshold":
                img_path = os.path.join(DATASET_DIR, "otsu-thresholding", selected_sample, selected_image)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                self.axes[i].imshow(img, cmap="gray")
                self.axes[i].set_title(titles[i], fontsize=10, fontweight='bold')
            else:
                # Tampilkan layar kosong bertuliskan "Not Found" jika gambar tahap tersebut belum diproses
                self.axes[i].text(0.5, 0.5, "Belum Diproses /\nTidak Ditemukan", 
                                  ha='center', va='center', color='red', fontsize=10)
                self.axes[i].set_title(titles[i], fontsize=10, color='gray')
                
            self.axes[i].axis("off")
            
        self.fig.tight_layout()
        self.canvas.draw()

    def clear_plots(self, message):
        """Membersihkan layar jika terjadi eror data tidak ditemukan"""
        for ax in self.axes:
            ax.clear()
            ax.text(0.5, 0.5, message, ha='center', va='center', color='gray')
            ax.axis("off")
        self.canvas.draw()

# =====================================================================
# RUN APPLICATION
# =====================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetViewerApp(root)
    root.mainloop()