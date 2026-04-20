import os
import numpy as np
import random
from PIL import Image
from perceptron import Perceptron
from associator import denoise
import pickle
from config import (RANDOM_SEED, WIDTH, HEIGHT, INPUT_SIZE, EPOCHS, NOISE_PROBABILITY, TRAINING_DATA_FOLDER, 
                    MODEL_FILE)

# set random seeds for reproducible results
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_image(path):
    img = Image.open(path)
    img = img.convert("RGBA")

    if img.size != (WIDTH, HEIGHT):
        img = img.resize((WIDTH, HEIGHT), Image.NEAREST)

    data = np.array(img)
    alpha = data[:, :, 3]
    rgb = data[:, :, :3]
    intensity = np.mean(rgb, axis=2)
    binary = ((alpha > 0) & (intensity < 128)).astype(int)

    return binary.flatten()

def load_dataset(folder):
    X = []

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith((".png")):
            path = os.path.join(folder, filename)
            vector = load_image(path)
            X.append(vector)

    return np.array(X)

def add_noise(vec, prob=NOISE_PROBABILITY):
    noisy = vec.copy()
    flip = np.random.rand(len(noisy)) < prob
    noisy[flip] = 1 - noisy[flip]
    return noisy

if __name__ == "__main__":
    X = load_dataset(TRAINING_DATA_FOLDER)
    if len(X) == 0:
        raise SystemExit(f"No training images found in {TRAINING_DATA_FOLDER}")

    print(f"Loaded {len(X)} training samples from {TRAINING_DATA_FOLDER}")

    perceptrons = [Perceptron(INPUT_SIZE) for _ in range(INPUT_SIZE)]
    print(f"Training {INPUT_SIZE} output perceptrons for {EPOCHS} epochs")

    X_noisy_eval = np.array([add_noise(x) for x in X])

    for epoch in range(1, EPOCHS + 1):
        X_noisy_train = np.array([add_noise(x) for x in X])
        
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        X_noisy_shuffled = X_noisy_train[indices]

        for x_noisy, x_clean in zip(X_noisy_shuffled, X_shuffled):
            for idx, perceptron in enumerate(perceptrons):
                perceptron.train(x_noisy, x_clean[idx])

        for idx, perceptron in enumerate(perceptrons):
            targets = X[:, idx]
            perceptron.update_pocket(X_noisy_eval, targets)

        if epoch % 10 == 0 or epoch == EPOCHS:
            eval_preds = np.array([denoise(perceptrons, x_noisy) for x_noisy in X_noisy_eval])
            eval_pixel_acc = np.mean(eval_preds == X)
            eval_sample_acc = np.mean(np.all(eval_preds == X, axis=1))
            print(f"Epoch {epoch}/{EPOCHS} | pixel acc: {eval_pixel_acc:.2%} | full-image acc: {eval_sample_acc:.2%}")

    for perceptron in perceptrons:
        perceptron.restore_best()

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(perceptrons, f)

    noisy_outputs = np.array([denoise(perceptrons, add_noise(x)) for x in X])
    noisy_pixel_accuracy = np.mean(noisy_outputs == X)
    noisy_sample_accuracy = np.mean(np.all(noisy_outputs == X, axis=1))
    pixel_errors = np.sum(noisy_outputs != X, axis=1)
    mean_pixel_errors = np.mean(pixel_errors)
    median_pixel_errors = np.median(pixel_errors)
    small_error_rate = np.mean(pixel_errors <= 5)

    print(f"Noisy pixel accuracy: {noisy_pixel_accuracy:.2%}")
    print(f"Noisy full-image accuracy: {noisy_sample_accuracy:.2%}")
    print(f"Mean pixel errors per image: {mean_pixel_errors:.2f}")
    print(f"Median pixel errors per image: {median_pixel_errors:.0f}")
    print(f"Images with <= 5 pixel errors: {small_error_rate:.2%}")
    print(f"Saved trained model to {MODEL_FILE}")