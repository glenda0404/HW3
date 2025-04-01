import numpy as np
import math
from scipy.interpolate import lagrange

def lagrange_interpolation(x_values, y_values, x_target):
    """
    使用 SciPy 的 lagrange 函式做拉格朗日插值，
    回傳在 x_target 的近似值。
    """
    poly = lagrange(x_values, y_values)
    return poly(x_target)

def error_bound(x_values, x_target):
    """
    使用插值誤差公式的上界：
        E_n(x) <= ( max|f^(n+1)(ξ)| ) * Π|x - x_i| / (n+1)!
    理論上 max|f^(n+1)(ξ)| 是在包含所有插值點與目標點的區間內 f^(n+1)(x) 的最大值，
    但對於 f(x)=cos(x) 而言，cos 或 sin 的絕對值最大均不超過 1，
    故直接取 1 作為保守估計。
    """
    n = len(x_values) - 1  # n 次多項式使用了 n+1 個點
    factorial_term = math.factorial(n+1)
    product_term = np.prod([abs(x_target - x) for x in x_values])
    max_derivative = 1  # 保守上界
    return (max_derivative * product_term) / factorial_term

# 題目給定的 4 個插值點
x_vals = np.array([0.698, 0.733, 0.768, 0.803])
y_vals = np.array([0.7661, 0.7432, 0.7193, 0.6946])

# 要插值的目標點
x_target = 0.750

# 計算實際 cos(0.750)，供參考
true_val = math.cos(0.750)
print(f"實際 cos(0.750) = {true_val:.7f}\n")

# 進行 1, 2, 3 次插值 (點數足夠)
for degree in [1, 2, 3]:
    # 取前 (degree+1) 個點來構造 degree 次插值多項式
    interp_x_vals = x_vals[:degree+1]
    interp_y_vals = y_vals[:degree+1]

    approx_value = lagrange_interpolation(interp_x_vals, interp_y_vals, x_target)
    err_bound = error_bound(interp_x_vals, x_target)
    actual_error = abs(approx_value - true_val)

    print(f"{degree} 次插值：")
    print(f"  近似值         = {approx_value:.7f}")
    print(f"  誤差上界       = {err_bound:.7e}")
    print(f"  真實誤差(參考) = {actual_error:.7e}\n")

# 額外提示，由於給定點不足，無法進行四次插值
print("注意：給定的點數不足，無法進行四次插值。")
