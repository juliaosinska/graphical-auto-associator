import numpy as np

def denoise(perceptrons, x):
    return np.array([p.predict(x) for p in perceptrons], dtype=int)

def denoise_iterative(perceptrons, x, steps):
    history = [x.astype(int)]
    current = x.astype(int)

    for _ in range(steps):
        next_state = denoise(perceptrons, current)
        history.append(next_state)
        if np.array_equal(next_state, current):
            break
        current = next_state

    return history