#Стохастический градиентный спуск (SGD — Stochastic Gradient Descent)

import numpy as np
import matplotlib.pyplot as plt

# =============================================
# Исходные данные
# =============================================
streetlights = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 1]
])

walk_vs_stop = np.array([[0], [1], [0], [1], [1], [0]])
weights = np.array([0.5, 0.48, -0.7])
learning_rate = 0.1
iterations = 10

# =============================================
# Обучение модели
# =============================================
error_history = []          # Суммарные ошибки по эпохам
step_error_history = []     # Ошибки на каждом шаге внутри эпох
weight_history = []         # История весов

print("🚦 Начинаем обучение модели...")
print(f"🔹 Начальные веса: {weights}")
print("-" * 50)

for epoch in range(iterations):
    total_error = 0
    epoch_errors = []
    
    print(f"\n🔵 Эпоха {epoch + 1}/{iterations}")
    
    for i in range(len(streetlights)):
        input_data = streetlights[i]
        target = walk_vs_stop[i]
        
        prediction = input_data.dot(weights)
        error = (target - prediction) ** 2
        total_error += error
        epoch_errors.append(error[0])
        
        delta = prediction - target
        gradient = delta * input_data
        weights = weights - learning_rate * gradient
        
        print(f"Пример {i+1}: Ошибка = {error[0]:.4f}")
    
    error_history.append(total_error[0])
    step_error_history.append(epoch_errors)
    weight_history.append(weights.copy())
    
    print(f"📊 Суммарная ошибка эпохи: {total_error[0]:.5f}")

# =============================================
# Визуализация (3 графика в одной строке)
# =============================================
plt.figure(figsize=(18, 5))

# График 1: Общая ошибка по эпохам
plt.subplot(1, 3, 1)
plt.plot(error_history, 'r-o', linewidth=2)
plt.title("Суммарная ошибка по эпохам", fontsize=12)
plt.xlabel("Номер эпохи", fontsize=10)
plt.ylabel("Ошибка (MSE)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# График 2: Ошибки внутри эпох
plt.subplot(1, 3, 2)
for i, errors in enumerate(step_error_history):
    plt.plot(range(1, len(streetlights) + 1), errors, '-o', label=f'Эпоха {i+1}')
plt.title("Ошибки на примерах внутри эпох", fontsize=12)
plt.xlabel("Номер примера", fontsize=10)
plt.ylabel("Ошибка (MSE)", fontsize=10)
plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1))
plt.grid(True, linestyle='--', alpha=0.7)

# График 3: Динамика весов
plt.subplot(1, 3, 3)
weight_history = np.array(weight_history)
plt.plot(weight_history[:, 0], 'b-', label='Красный', linewidth=2)
plt.plot(weight_history[:, 1], 'g-', label='Желтый', linewidth=2)
plt.plot(weight_history[:, 2], 'k-', label='Зеленый', linewidth=2)
plt.title("Динамика изменения весов", fontsize=12)
plt.xlabel("Номер эпохи", fontsize=10)
plt.ylabel("Значение веса", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# =============================================
# Финальные результаты
# =============================================
print("\n🎯 Результаты обучения:")
print(f"Финальные веса: {np.round(weights, 4)}")
print("\n🧠 Интерпретация:")
print(f"• Красный свет: {'Важен' if weights[0] > 0.3 else 'Не важен'}")
print(f"• Желтый свет: {'Важен' if weights[1] > 0.3 else 'Не важен'}")
print(f"• Зеленый свет: {'Важен' if weights[2] > 0.3 else 'Не важен'}")