import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

# Вхідні дані
price = np.array([1.23, 1.55, 1.78, 2.04, 2.38, 2.80, 3.13, 3.52, 3.78, 4.20, 4.61, 5.00])
demand = np.array([121, 101, 93, 81, 75, 69, 61, 54, 50, 48, 47, 42])
supply = np.array([17, 31, 41, 47, 56, 61, 65, 68, 73, 75, 77, 80])

# Функція для підбору поліноміального рівняння
def poly_fit(x, a, b, c):
    return a * x**2 + b * x + c

# Підгонка для попиту
demand_params, _ = curve_fit(poly_fit, price, demand)

# Підгонка для пропозиції
supply_params, _ = curve_fit(poly_fit, price, supply)

# Функція для визначення точки рівноваги
def equilibrium(p, subsidy=0):
    return poly_fit(p - subsidy, *demand_params) - poly_fit(p, *supply_params)

# Розв'язок рівняння для визначення ціни рівноваги
price_eq = fsolve(equilibrium, 2.5)[0]  # Початкове наближення ціни

# Обчислення кількості рівноваги
quantity_eq = poly_fit(price_eq, *demand_params)

subsidy = 0.8
new_price_eq = fsolve(lambda p: equilibrium(p, subsidy), 2.5)[0]
new_quantity_eq = poly_fit(new_price_eq, *demand_params)

# Функція для розрахунку дугової еластичності
def arc_elasticity(q1, q2, p1, p2):
    dq = (q2 - q1) / ((q2 + q1) / 2)
    dp = (p2 - p1) / ((p2 + p1) / 2)
    return dq / dp

# Розрахунок дугової еластичності попиту та пропозиції
demand_elasticity = arc_elasticity(demand[-1], demand[0], price[-1], price[0])
supply_elasticity = arc_elasticity(supply[-1], supply[0], price[-1], price[0])

# Побудова графіків
prices_plot = np.linspace(price.min(), price.max(), 100)
demand_plot = poly_fit(prices_plot, *demand_params)
supply_plot = poly_fit(prices_plot, *supply_params)
new_supply_plot = poly_fit(prices_plot - subsidy, *supply_params)  # Пропозиція з урахуванням дотації

plt.figure(figsize=(10, 6))
plt.plot(prices_plot, demand_plot, label='Попит', color='blue')
plt.plot(prices_plot, supply_plot, label='Пропозиція', color='red')
plt.scatter(price, demand, color='blue', marker='o', label='Реальний попит')
plt.scatter(price, supply, color='red', marker='x', label='Реальна пропозиція')
plt.plot(price_eq, quantity_eq, 'go', label='Точка рівноваги')
plt.title('Функції попиту та пропозиції')
plt.xlabel('Ціна')
plt.ylabel('Кількість')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(prices_plot, demand_plot, label='Попит', color='blue')
plt.plot(prices_plot, supply_plot, label='Пропозиція без дотації', color='red', linestyle='--')
plt.plot(prices_plot, new_supply_plot, label='Пропозиція з дотацією', color='green')
plt.scatter(price_eq, quantity_eq, color='black', zorder=5, label='Точка рівноваги без дотації')
plt.scatter(new_price_eq, new_quantity_eq, color='orange', zorder=5, label='Точка рівноваги з дотацією')
plt.title('Зміна ринкової рівноваги після введення дотації')
plt.xlabel('Ціна')
plt.ylabel('Кількість')
plt.legend()
plt.grid(True)
plt.show()

# Дослідження стабільності рівноваги
demand_slope = 2 * demand_params[0] * price_eq + demand_params[1]
supply_slope = 2 * supply_params[0] * price_eq + supply_params[1]
print("Нахил кривої попиту в точці рівноваги:", demand_slope)
print("Нахил кривої пропозиції в точці рівноваги:", supply_slope)
if supply_slope > demand_slope:
    print("Стан рівноваги стабільний.")
else:
    print("Стан рівноваги нестабільний.")

# Виведення функцій попиту та пропозиції
print(f"Функція попиту: Q_d = {demand_params[0]:.2f}P^2 + {demand_params[1]:.2f}P + {demand_params[2]:.2f}")
print(f"Функція пропозиції: Q_s = {supply_params[0]:.2f}P^2 + {supply_params[1]:.2f}P + {supply_params[2]:.2f}")

# Виведення результатів
print(f"Точка рівноваги: Ціна = {price_eq:.2f}, Кількість = {quantity_eq:.2f}")
print(f"Дугова еластичність попиту: {demand_elasticity:.2f}")
print(f"Дугова еластичність пропозиції: {supply_elasticity:.2f}")
