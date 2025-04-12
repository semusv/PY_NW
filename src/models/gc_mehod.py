import numpy as np
import matplotlib.pyplot as plt

# Инициализация параметров
weight = 0.5        # Начальное значение веса (параметра модели)
input = 0.5         # Фиксированный вход сети (в реальных задачах это данные)
goal_prediction = 0.8  # Целевое значение, которое модель должна предсказать
alpha = 0.01         # Скорость обучения (learning rate) - шаг градиентного спуска

# Массивы для записи истории обучения (используются для графиков)
weights_arr = []     # История значений веса
prediction_arr = []  # История предсказаний
errors_arr = []      # История ошибок (MSE)

# Основной цикл обучения (2000 итераций градиентного спуска)
for iteration in range(2000):
    ### 1. Прямое распространение (forward pass) ###
    prediction = input * weight  # Линейная модель: y = x * w
    
    ### 2. Вычисление ошибки ###
    # Mean Squared Error (MSE) - квадрат разницы между предсказанием и целью
    error = (prediction - goal_prediction) ** 2
    
    # Сохраняем значения для визуализации
    errors_arr.append(error)
    weights_arr.append(weight)
    prediction_arr.append(prediction)
    
    ### 3. Обратное распространение (backpropagation) ###
    # Вычисляем производные для градиентного спуска:
    
    # Производная ошибки по предсказанию (dE/dy):
    # E = (y - y_goal)^2  =>  dE/dy = 2*(y - y_goal)
    # (Коэффициент 2 часто опускают, объединяя с learning rate)
    delta = prediction - goal_prediction  # Упрощённая версия dE/dy
    
    # Производная предсказания по весу (dy/dw):
    # y = x * w  =>  dy/dw = x
    derivative_prediction_wrt_weight = input
    
    # Полная производная ошибки по весу (dE/dw) по правилу цепи:
    # dE/dw = (dE/dy) * (dy/dw) = delta * x
    weight_delta = delta * input
    
    ### 4. Обновление веса (градиентный спуск) ###
    # Новый вес = старый вес - learning_rate * градиент
    weight = weight - weight_delta * alpha

# Подготовка данных для графиков
iterations = np.arange(len(weights_arr))  # Номера итераций

# Создаём фигуру с двумя графиками
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

### График 1: Динамика веса и предсказания ###
ax1.plot(iterations, weights_arr, label='Weight (вес)')
ax1.plot(iterations, prediction_arr, label='Prediction (предсказание)')
ax1.axhline(goal_prediction, color='r', linestyle='--', label='Цель')
ax1.set_xlabel('Итерация')
ax1.set_ylabel('Значение')
ax1.legend()
ax1.grid(True)
ax1.set_title('Обучение линейной модели')

### График 2: Зависимость ошибки от веса ###
ax2.plot(weights_arr, errors_arr, label='Error (ошибка)', color='green')
ax2.set_xlabel('Weight (вес)')
ax2.set_ylabel('MSE Error')
ax2.legend()
ax2.grid(True)
ax2.set_title('Парабола ошибки')

plt.tight_layout()
plt.show()