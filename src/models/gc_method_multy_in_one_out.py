#Множественный вход с одним выходом и градиентным спуском
import numpy as np
import matplotlib.pyplot as plt

def w_sum(a,b):
    assert(len(a)==len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i]*b[i])
    return output

def neural_network(input,weights):
    pred = w_sum(input,weights)
    return pred

def ele_mul(number,vector):
    output = np.zeros(3)
    assert(len(output) ==len(vector))
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output

#Начальные веса
weights = [0.1,0.2,-0.1]
#Начальные парамерты входные
toes =  [8.50, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.20, 1.3, 0.5, 1.0]
#Результа
win_or_lose_binary = [1,1,0,1]
alpha = 0.01

true = win_or_lose_binary[0]
input = [toes[0],wlrec[0],nfans[0]]


print(f"weights {weights}")

for iter in range(300):
    #работаем с ошибкой предсказания
    pred = neural_network(input,weights)
    error = (pred - true) ** 2
    delta = pred - true

    if error == 0:
        break

    #Находим производные(приращение)
    weights_deltas = ele_mul(delta,input)

    for i in range(len(weights)):
        weights[i] -= alpha * weights_deltas[i]

    print(f"----------Iteration-- {iter}")
    print(f"Pred {pred:.3f}")
    print(f"Error {error:.50f}")
    print(f"Delta {delta:.3f}")
    print(f"weights_deltas: {[f'{x:.2f}' for x in weights_deltas]}")
    print(f"weights: {[f'{x:.4f}' for x in weights]}")





# # ... (Ваш исходный код без изменений до визуализации) ...

# # Функция для вычисления ошибки при изменении одного веса
# def calculate_single_weight_error(weight_idx, weight_values, fixed_weights, input_data, true_value):
#     errors = []
#     for w in weight_values:
#         temp_weights = fixed_weights.copy()
#         temp_weights[weight_idx] = w
#         pred = neural_network(input_data, temp_weights)
#         errors.append((pred - true_value)**2)
#     return errors

# # Подготовка данных для графиков
# weight_names = ['Вес для toes', 'Вес для wlrec', 'Вес для nfans']
# input_names = ['toes', 'wlrec', 'nfans']

# # Создаем 3 субплога (по одному на каждый вес)
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # Для каждого веса строим свой график
# for i in range(3):
#     # Генерируем диапазон значений для текущего веса
#     weight_range = np.linspace(weights[i] - 0.3, weights[i] + 0.3, 100)
    
#     # Вычисляем ошибки для этого диапазона
#     errors = calculate_single_weight_error(i, weight_range, weights, input, true)
    
#     # Строим график
#     ax = axes[i]
#     ax.plot(weight_range, errors, label='Ошибка', linewidth=2)
#     ax.axvline(weights[i], color='r', linestyle='--', label='Текущий вес')
    
#     # Отмечаем точку минимума
#     min_idx = np.argmin(errors)
#     ax.scatter(weight_range[min_idx], errors[min_idx], color='g', s=100, 
#                label=f'Минимум: {weight_range[min_idx]:.2f}')
    
#     # Настраиваем оформление
#     ax.set_xlabel(f'Значение веса {i+1} ({input_names[i]})')
#     ax.set_ylabel('Ошибка MSE')
#     ax.set_title(f'Зависимость ошибки от {weight_names[i]}')
#     ax.grid(True)
#     ax.legend()

# plt.tight_layout()
# plt.show()

# # Вывод численных результатов (как в вашем коде)
# print(f"\nPred: {pred:.3f}")
# print(f"Error: {error:.3f}")
# print(f"Delta: {delta:.3f}")
# print(f"Weights deltas: {[f'{x:.3f}' for x in weights_deltas]}")
# print(f"Updated weights: {[f'{x:.4f}' for x in weights]}")