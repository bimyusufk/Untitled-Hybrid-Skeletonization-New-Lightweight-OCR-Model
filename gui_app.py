import os
import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from super_hybrid_benchmarking import (
    TopoGradNet,
    preprocess_image,
    extract_super_features,
    CLASS_LIST,
)

# Configuration
MODEL_PATH = "ocr_evaluation_outputs_super_hybrid/TopoGrad-Net.pth"

class TopoGradOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TopoGrad-Net OCR Client")
        self.root.geometry("900x650")
        self.root.configure(bg="#1e1e1e")
        
        # Load Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        
        # Create UI Layout
        self.setup_ui()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showwarning(
                "Model Warning", 
                f"Model weights not found at: {MODEL_PATH}\nInference will run in demo mode if possible."
            )
            return None
        try:
            model = TopoGradNet(num_classes=62, feat_dim=12)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {e}")
            return None

    def setup_ui(self):
        # Header Style
        header = tk.Label(
            self.root, 
            text="🔮 TopoGrad-Net OCR Client", 
            font=("Helvetica", 16, "bold"), 
            bg="#1e1e1e", 
            fg="#00e676"
        )
        header.pack(pady=10)
        
        # Button for Upload
        btn_upload = tk.Button(
            self.root, 
            text="Open Image File", 
            command=self.upload_and_predict,
            font=("Helvetica", 11, "bold"),
            bg="#6200ee", 
            fg="white", 
            activebackground="#3700b3", 
            activeforeground="white",
            relief="flat",
            padx=15, 
            pady=8
        )
        btn_upload.pack(pady=10)

        # Main frame for visual elements
        self.main_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Grid layout for images (Col 1: Raw, Col 2: Preprocess, Col 3: Gradient)
        self.img_frame = tk.LabelFrame(
            self.main_frame, 
            text=" Preprocessing Pipeline Stages ", 
            font=("Helvetica", 10, "bold"),
            bg="#1e1e1e", 
            fg="#e6e6fa",
            relief="groove", 
            bd=2
        )
        self.img_frame.pack(fill="x", pady=10)
        
        # Sub-frames for the three stages
        self.create_image_slot(self.img_frame, 0, "1. Raw Image")
        self.create_image_slot(self.img_frame, 1, "2. Otsu + Hole Filling")
        self.create_image_slot(self.img_frame, 2, "3. Morphological Gradient")

        # Bottom section: Left for Prediction result, Right for Feature vector
        self.details_frame = tk.Frame(self.main_frame, bg="#1e1e1e")
        self.details_frame.pack(fill="both", expand=True, pady=10)
        
        # Predictions (Left)
        self.pred_frame = tk.LabelFrame(
            self.details_frame, 
            text=" Prediction Output ", 
            font=("Helvetica", 10, "bold"),
            bg="#1e1e1e", 
            fg="#e6e6fa",
            relief="groove", 
            bd=2,
            width=300
        )
        self.pred_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.char_label = tk.Label(
            self.pred_frame, 
            text="-", 
            font=("Helvetica", 64, "bold"), 
            bg="#1e1e1e", 
            fg="#00e676"
        )
        self.char_label.pack(pady=15)
        
        self.conf_label = tk.Label(
            self.pred_frame, 
            text="Confidence: -", 
            font=("Helvetica", 12), 
            bg="#1e1e1e", 
            fg="#a5a5c5"
        )
        self.conf_label.pack(pady=5)
        
        # Feature vector (Right)
        self.feat_frame = tk.LabelFrame(
            self.details_frame, 
            text=" Extracted Topological Features (12-dim) ", 
            font=("Helvetica", 10, "bold"),
            bg="#1e1e1e", 
            fg="#e6e6fa",
            relief="groove", 
            bd=2
        )
        self.feat_frame.pack(side="right", fill="both", expand=True)
        
        # Scrollable list for features
        self.tree = ttk.Treeview(self.feat_frame, columns=("Feature", "Value"), show="headings", height=8)
        self.tree.heading("Feature", text="Feature Name")
        self.tree.heading("Value", text="Value")
        self.tree.column("Feature", width=220, anchor="w")
        self.tree.column("Value", width=120, anchor="e")
        self.tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        scroll = ttk.Scrollbar(self.feat_frame, orient="vertical", command=self.tree.yview)
        scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scroll.set)

    def create_image_slot(self, parent, col, title):
        frame = tk.Frame(parent, bg="#1e1e1e", padx=10, pady=5)
        frame.grid(row=0, column=col, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)
        
        lbl_title = tk.Label(frame, text=title, font=("Helvetica", 9, "bold"), bg="#1e1e1e", fg="#a5a5c5")
        lbl_title.pack(anchor="center", pady=(0, 5))
        
        # Placeholder canvas
        canvas = tk.Canvas(frame, width=200, height=200, bg="#2a2a2a", highlightthickness=0)
        canvas.pack(anchor="center")
        canvas.create_text(100, 100, text="No Image", fill="#555555", font=("Helvetica", 10))
        
        # Save reference
        if not hasattr(self, 'slots'):
            self.slots = {}
        self.slots[col] = canvas

    def upload_and_predict(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not file_path:
            return
            
        try:
            # 1. Processing pipeline
            img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise ValueError("Could not read image file.")
                
            img_bin, _ = preprocess_image(file_path)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img_grad = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel)
            
            feats = extract_super_features(img_bin)
            
            # Update GUI Images
            self.display_image_in_slot(0, img_gray)
            self.display_image_in_slot(1, img_bin)
            self.display_image_in_slot(2, img_grad)
            
            # Predict
            if self.model is not None:
                img_norm = (img_grad.astype(np.float32) / 255.0 - 0.5) / 0.5
                img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
                feats_tensor = torch.tensor(feats).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(img_tensor, feats_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    
                self.char_label.config(text=CLASS_LIST[predicted.item()])
                self.conf_label.config(text=f"Confidence: {confidence.item() * 100:.2f}%")
            else:
                self.char_label.config(text="N/A")
                self.conf_label.config(text="Model Weights Missing")
                
            # Update Features Tree
            self.update_features_table(feats)
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {e}")

    def display_image_in_slot(self, col, cv_img):
        # Resize to fit 200x200 canvas
        h, w = cv_img.shape[:2]
        scale = min(200.0/w, 200.0/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        cv_img_resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        pil_img = Image.fromarray(cv_img_resized)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        canvas = self.slots[col]
        canvas.delete("all")
        # Center image on canvas
        offset_x = (200 - new_w) // 2
        offset_y = (200 - new_h) // 2
        canvas.create_image(offset_x, offset_y, anchor="nw", image=tk_img)
        canvas.image = tk_img # Keep reference

    def update_features_table(self, feats):
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        feature_names = [
            "Euler Number", "Eccentricity", "Aspect Ratio", "Extent", "Solidity",
            "Hu Moment 1", "Hu Moment 2", "Hu Moment 3", "Hu Moment 4", "Hu Moment 5", "Hu Moment 6", "Hu Moment 7"
        ]
        
        for name, val in zip(feature_names, feats):
            self.tree.insert("", "end", values=(name, f"{val:.5f}"))

if __name__ == "__main__":
    # Configure custom style for treeview to match dark theme
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("default")
    style.configure(
        "Treeview", 
        background="#2a2a2a", 
        foreground="#e6e6fa", 
        fieldbackground="#2a2a2a",
        rowheight=22
    )
    style.map("Treeview", background=[("selected", "#6200ee")])
    style.configure("Treeview.Heading", background="#333333", foreground="white", relief="flat")
    
    app = TopoGradOCRApp(root)
    root.mainloop()
