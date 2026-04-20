import os
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from associator import denoise, denoise_iterative
from config import WIDTH, HEIGHT, PIXEL_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT, MODEL_FILE, NOISE_PROBABILITY, MAX_ITER_STEPS

CANVAS_WIDTH = WIDTH * PIXEL_SIZE
CANVAS_HEIGHT = HEIGHT * PIXEL_SIZE

def binary_image_to_photo(array):
    image = Image.fromarray((array.astype(np.uint8) * 255), mode="L")
    image = image.resize((CANVAS_WIDTH, CANVAS_HEIGHT), Image.NEAREST)
    return ImageTk.PhotoImage(image)

def add_noise_to_array(array, probability):
    noisy = array.copy().flatten()
    mask = np.random.rand(noisy.size) < probability
    noisy[mask] = 1 - noisy[mask]
    return noisy.reshape(array.shape)

class AutoassociatorApp:
    def __init__(self, root, perceptrons):
        self.root = root
        self.perceptrons = perceptrons
        self.root.title("Autoasocjator obrazu - odszumianie")
        self.root.configure(bg="#f0f0f0")

        title_label = tk.Label(root, text="Autoasocjator dla czarno-białych obrazów", font=("Arial", 16, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)

        control_frame = tk.Frame(root, bg="#f0f0f0")
        control_frame.pack(pady=6)

        load_button = tk.Button(control_frame, text="Wczytaj obraz", width=14, command=self.load_image)
        load_button.grid(row=0, column=0, padx=5, pady=4)

        noise_button = tk.Button(control_frame, text="Dodaj szum", width=14, command=self.auto_noise)
        noise_button.grid(row=0, column=1, padx=5, pady=4)

        denoise_button = tk.Button(control_frame, text="Odszumuj", width=14, command=self.denoise_image)
        denoise_button.grid(row=0, column=2, padx=5, pady=4)

        iterate_button = tk.Button(control_frame, text="Iteracyjne odszumianie", width=18, command=self.iterative_denoise)
        iterate_button.grid(row=0, column=3, padx=5, pady=4)

        reset_button = tk.Button(control_frame, text="Resetuj obrazy", width=14, command=self.reset_images)
        reset_button.grid(row=0, column=4, padx=5, pady=4)

        slider_label = tk.Label(control_frame, text="Prawdopodobieństwo szumu:", bg="#f0f0f0")
        slider_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=5)

        self.noise_scale = tk.Scale(control_frame, from_=0.0, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, length=280)
        self.noise_scale.set(NOISE_PROBABILITY)
        self.noise_scale.grid(row=1, column=2, columnspan=3, padx=5)

        self.status_label = tk.Label(root, text="Wczytaj obraz, dodaj szum i uruchom odszumianie.", bg="#f0f0f0", fg="#333", font=("Arial", 12))
        self.status_label.pack(pady=4)

        display_frame = tk.Frame(root, bg="#f0f0f0")
        display_frame.pack(pady=6)

        self.original_canvas = tk.Canvas(display_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white", bd=2, relief="sunken")
        self.noisy_canvas = tk.Canvas(display_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white", bd=2, relief="sunken")
        self.denoised_canvas = tk.Canvas(display_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white", bd=2, relief="sunken")
        self.iterated_canvas = tk.Canvas(display_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white", bd=2, relief="sunken")

        self.original_canvas.grid(row=0, column=0, padx=6, pady=6)
        self.noisy_canvas.grid(row=0, column=1, padx=6, pady=6)
        self.denoised_canvas.grid(row=1, column=0, padx=6, pady=6)
        self.iterated_canvas.grid(row=1, column=1, padx=6, pady=6)

        labels_frame = tk.Frame(root, bg="#f0f0f0")
        labels_frame.pack(pady=2)

        tk.Label(labels_frame, text="Oryginał", bg="#f0f0f0").grid(row=0, column=0, padx=20)
        tk.Label(labels_frame, text="Zaszumiony", bg="#f0f0f0").grid(row=0, column=1, padx=20)
        tk.Label(labels_frame, text="Odszumiony", bg="#f0f0f0").grid(row=1, column=0, padx=20)
        tk.Label(labels_frame, text="Iteracja", bg="#f0f0f0").grid(row=1, column=1, padx=20)

        self.original_image_id = self.original_canvas.create_image(0, 0, anchor="nw")
        self.noisy_image_id = self.noisy_canvas.create_image(0, 0, anchor="nw")
        self.denoised_image_id = self.denoised_canvas.create_image(0, 0, anchor="nw")
        self.iterated_image_id = self.iterated_canvas.create_image(0, 0, anchor="nw")

        self.original_photo = None
        self.noisy_photo = None
        self.denoised_photo = None
        self.iterated_photo = None

        self.original = np.zeros((HEIGHT, WIDTH), dtype=int)
        self.noisy = self.original.copy()
        self.denoised = self.original.copy()
        self.iterated = self.original.copy()

        self.noisy_canvas.bind("<Button-1>", self.toggle_noisy_pixel)
        self.update_canvases()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Wczytaj obraz",
            filetypes=[("Obrazy", "*.png")],
        )
        if not file_path:
            return

        image = Image.open(file_path).convert("RGBA")
        if image.size != (WIDTH, HEIGHT):
            image = image.resize((WIDTH, HEIGHT), Image.NEAREST)

        data = np.array(image)
        alpha = data[:, :, 3]
        rgb = data[:, :, :3]
        intensity = np.mean(rgb, axis=2)
        self.original = ((alpha > 0) & (intensity < 128)).astype(int)
        self.noisy = self.original.copy()
        self.denoised = self.original.copy()
        self.iterated = self.original.copy()
        self.update_canvases()
        self.status_label.config(text=f"Wczytano obraz {os.path.basename(file_path)}.")

    def auto_noise(self):
        probability = self.noise_scale.get()
        self.noisy = add_noise_to_array(self.original, probability)
        self.denoised = self.original.copy()
        self.iterated = self.original.copy()
        self.update_canvases()
        self.status_label.config(text=f"Dodano szum o prawdopodobieństwie {probability:.2f}.")

    def toggle_noisy_pixel(self, event):
        x = min(max(event.x // PIXEL_SIZE, 0), WIDTH - 1)
        y = min(max(event.y // PIXEL_SIZE, 0), HEIGHT - 1)
        self.noisy[y, x] = 1 - self.noisy[y, x]
        self.denoised = self.original.copy()
        self.iterated = self.original.copy()
        self.update_canvases()
        self.status_label.config(text=f"Ręcznie zmieniono piksel ({x}, {y}).")

    def denoise_image(self):
        vector = self.noisy.flatten()
        output = denoise(self.perceptrons, vector)
        self.denoised = output.reshape((HEIGHT, WIDTH))
        self.iterated = self.denoised.copy()
        self.update_canvases()
        self.status_label.config(text="Wykonano jedno odszumienie.")

    def iterative_denoise(self):
        vector = self.noisy.flatten()
        history = denoise_iterative(self.perceptrons, vector, steps=MAX_ITER_STEPS)
        self.denoised = history[1].reshape((HEIGHT, WIDTH))
        self.iterated = history[-1].reshape((HEIGHT, WIDTH))
        stable = np.array_equal(history[-1], history[-2])
        stability_text = "stabilne" if stable else "niestabilne"
        self.update_canvases()
        self.status_label.config(text=f"Iteracyjne odszumianie ({MAX_ITER_STEPS} kroków): {stability_text}.")

    def reset_images(self):
        self.original = np.zeros((HEIGHT, WIDTH), dtype=int)
        self.noisy = self.original.copy()
        self.denoised = self.original.copy()
        self.iterated = self.original.copy()
        self.update_canvases()
        self.status_label.config(text="Reset obrazów.")

    def update_canvases(self):
        self.original_photo = binary_image_to_photo(self.original)
        self.noisy_photo = binary_image_to_photo(self.noisy)
        self.denoised_photo = binary_image_to_photo(self.denoised)
        self.iterated_photo = binary_image_to_photo(self.iterated)

        self.original_canvas.itemconfig(self.original_image_id, image=self.original_photo)
        self.noisy_canvas.itemconfig(self.noisy_image_id, image=self.noisy_photo)
        self.denoised_canvas.itemconfig(self.denoised_image_id, image=self.denoised_photo)
        self.iterated_canvas.itemconfig(self.iterated_image_id, image=self.iterated_photo)

try:
    with open(MODEL_FILE, "rb") as f:
        perceptrons = pickle.load(f)
except FileNotFoundError:
    print(f"Error: {MODEL_FILE} not found. Run train.py first.")
    perceptrons = None

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    root.resizable(False, False)

    if perceptrons is not None:
        app = AutoassociatorApp(root, perceptrons)
        root.mainloop()
    else:
        print("Cannot start GUI: model not loaded")