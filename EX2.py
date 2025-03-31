import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# 數據點 (x -> y = e^(-x))
x_vals = np.array([0.3, 0.4, 0.5, 0.6])  # x 正數
y_vals = np.exp(-x_vals)  # y = e^(-x)

# 使用線性插值建立反函數
inverse_interp = interp1d(y_vals, x_vals, kind='linear', fill_value="extrapolate")

# 解法：解決 x = e^(-x) <=> inverse_interp(x) = x
def equation(x):
    return inverse_interp(x) - x

# 使用 fsolve 求解，初始猜測值設為 0.5
solution = fsolve(equation, 0.5)[0]

# 輸出結果
print(f"使用逆插值法找到的解: x ≈ {solution:.6f}")
