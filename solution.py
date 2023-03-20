from math import exp, sqrt, tan, atan, cos, sin, log
import matplotlib.pyplot as plt
import numpy as np

# Функции для тестирования метода Рунге
def f(x, y):
    return (x-x**2)*y
def ans(x):
    return exp(-1/6*x**2*(-3+2*x))
def f1(x, y): # тестовая функция 1
    return sin(x)-y

def f2(x, y): # тестовая функция 2
    return y*cos(x)+sin(2*x)

def f31(x, u, v): # тестовая функция 3_1
    return cos(x+1.1*v)+u

def f32(x, u, v): # тестовая функция 3_2
    return -v**2+2.1*u+1.1

def f41(x, u, v): # тестовая функция 4_1
    return -v+sin(x)

def f42(x, u, v): # тестовая функция 4_2
    return u+cos(x)

def ans1(x): # точное решение для функции 1
    return -0.5*cos(x)+0.5*sin(x)+21/2*exp(-x)

def ans2(x): # точное решение для функции 2
    return -2*sin(x)+2*exp(sin(x))-2

def ans4_1(x): # точное решение u для системы 4
    return sin(x)+cos(x)

def ans4_2(x): # точное решение v для системы 4
    return 2*sin(x)-cos(x)

# Функции для тестирования решения краевой задачи
# Функции для теста 1
def p(x): 
    return 1.5

def q(x):
    return -x

def f(x):
    return 0.5

def p1(x): 
    return 2*x**2

def q1(x):
    return 1

def f1(x):
    return x

# Функции для теста 2
def p2(x):
    return 0

def q2(x):
    return 1

def f2(x):
    return x+2*exp(x)

def ans22(x):
    return -sin(x)+cos(x)+exp(x)+x

# Функции для теста 3
def p3(x):
    return -3

def q3(x):
    return 2

def f3(x):
    return sin(x)

def ans23(x):
    return sin(x)/10+3*cos(x)/10+exp(x)+exp(2*x)

# Функции для теста 3
def p4(x):
    return 1/x

def q4(x):
    return 0

def f4(x):
    return 0

def ans24(x):
    return 6*log(x)+5

# Основные функции, осуществляющие вычисления
def arange(start, stop, n): 
    step = (stop - start) / n
    return [round(x*step, 10) for x in range(int(start/step), int(stop/step))]


def make_graphic(xmas, ymas, marker='', cl = None, lw = 1): #строит график
    return plt.plot(xmas, ymas, marker, color=cl, linewidth = lw)

def runge2(f, a, b, y0, n, xmas, ymas): # метод Рунге 2-го порядка для ОДУ
    xmas.clear()
    ymas.clear()
    d = (b-a)/n
    xmas.append(a)
    ymas.append(y0)
    for i in range(n):
        res = f(a, y0)
        y0 += d/2*(res+f(a+d, y0+res*d))
        a += d
        xmas.append(a)
        ymas.append(y0)
        
def runge4(f, a, b, y0, n, xmas, ymas): # метод Рунге 4-го порядка для ОДУ
    xmas.clear()
    ymas.clear()
    d = (b-a)/n
    xmas.append(a)
    ymas.append(y0)
    for i in range(n):
        res1 = f(a, y0)
        res2 = f(a+d/2, y0+d/2*res1)
        res3 = f(a+d/2, y0+d/2*res2)
        res4 = f(a+d, y0+d*res3)
        y0 += d/6*(res1+2*res2+2*res3+res4)
        a += d
        xmas.append(a)
        ymas.append(y0)

def runge2_sys(f1, f2, a, b, y01, y02, n, xmas, ymas1, ymas2): # метод Рунге 2-го порядка для системы ОДУ
    xmas.clear()
    ymas1.clear()
    ymas2.clear()
    d = (b-a)/n
    xmas.append(a)
    ymas1.append(y01)
    ymas2.append(y02)
    n -= 1
    for i in range(n):
        res1 = f1(a, y01, y02)
        res2 = f2(a, y01, y02)
        y01 += d/2*(res1+f1(a+d, y01+res1*d, y02+res2*d))
        y02 += d/2*(res2+f2(a+d, y01+res1*d, y02+res2*d))
        a += d
        xmas.append(a)
        ymas1.append(y01)
        ymas2.append(y02)

def runge4_sys(f1, f2, a, b, y01, y02, n, xmas, ymas1, ymas2): # метод Рунге 4-го порядка для системы ОДУ
    xmas.clear()
    ymas1.clear()
    ymas2.clear()
    d = (b-a)/n
    xmas.append(a)
    ymas1.append(y01)
    ymas2.append(y02)
    n -= 1
    for i in range(n):
        res11 = f1(a, y01, y02)
        res21 = f2(a, y01, y02)
        res12 = f1(a+d/2, y01+d/2*res11, y02+d/2*res21)
        res22 = f2(a+d/2, y01+d/2*res11, y02+d/2*res21)
        res13 = f1(a+d/2, y01+d/2*res12, y02+d/2*res22)
        res23 = f2(a+d/2, y01+d/2*res12, y02+d/2*res22)
        res14 = f1(a+d, y01+d*res13, y02+d*res23)
        res24 = f2(a+d, y01+d*res13, y02+d*res23)
        y01 += d/6*(res11+2*res12+2*res13+res14)
        y02 += d/6*(res21+2*res22+2*res23+res24)
        a += d
        xmas.append(a)
        ymas1.append(y01)
        ymas2.append(y02)

def run_through_method(a, b, p, q, f, n, sigma, gamma, delta): # Метод прогонки решения краевой задачи
    d = (b-a)/n
    n+=1
    amas = []
    bmas = []
    cmas = []
    fmas = []
    alpha = np.zeros(n-1)
    beta = np.zeros(n-1)
    result = np.zeros(n)
    def coef_a(m):
        return 1 - p(m)*d/2
    def coef_b(m):
        return 1 + p(m)*d/2
    def coef_c(m):
        return -2 + q(m)*(d**2)
    def coef_f(m):
        return f(m)*d**2

    
    for i in range(0,n):
        amas.append(coef_a(a+i*d))
        bmas.append(coef_b(a+i*d))
        cmas.append(coef_c(a+i*d))
        fmas.append(coef_f(a+i*d))
    amas[n-1] = -gamma[1]/d
    bmas[0] = gamma[0]/d
    cmas[0] = sigma[0]-gamma[0]/d
    cmas[n-1] = sigma[1]+gamma[1]/d
    fmas[0] = delta[0]
    fmas[n-1] = delta[1]
    alpha[0] = -bmas[0]/cmas[0]
    beta[0] = fmas[0]/cmas[0]

    for i in range(1, n-1):
        alpha[i] = -bmas[i]/(amas[i]*alpha[i-1]+cmas[i])
        beta[i] = (fmas[i] - amas[i] * beta[i-1])/(amas[i]*alpha[i-1]+cmas[i])

    result[n-1] = (fmas[n-1] - amas[n-1] * beta[n-2])/(amas[n-1]*alpha[n-2]+cmas[n-1])
    for i in range(n-2, -1, -1):
        result[i] = alpha[i]*result[i+1]+beta[i]
    return result

def boundary(p, q, f, sigma, gamma, delta, n, a = 1, b = 1.3): 
    h = (b-a)/n
    
    def A_func(x):
        return 1 - p(x)*h/2
    def C_func(x):
        return -2 + q(x)*(h**2)
    def B_func(x):
        return 1 + p(x)*h/2
    def F_func(x):
        return f(x)*(h**2)  
    
    # формирование коэффициентов трехдиагональной матрицы
    A = [A_func(a+i*h) for i in range(0, n+1)]
    B = [B_func(a+i*h) for i in range(0, n+1)]
    C = [C_func(a+i*h) for i in range(0, n+1)]
    F = [F_func(a+i*h) for i in range(0, n+1)]
    
    # задание краевых условий
    F[0] = delta[0]
    F[n] = delta[1]
    C[0] = sigma[0]-gamma[0]/h
    B[0] = gamma[0]/h
    A[n] = -gamma[1]/h
    C[n] = sigma[1]+gamma[1]/h
    
    # инициализация прогоночных коэффициентов
    alpha = [0]*n
    beta = [0]*n
    alpha[0] = -B[0]/C[0]
    beta[0] = F[0]/C[0]
    
    # рекуррентное вычисление всех прогоночных коэффициентов
    for i in range(1, n):
        alpha[i] = -B[i]/(A[i]*alpha[i-1]+C[i])
        beta[i] = (F[i]-A[i]*beta[i-1])/(C[i]+A[i]*alpha[i-1])
    
    # инициализация вектора неизвестных
    y = [0]*(n+1)
    y[n] = (F[n]-A[n]*beta[n-1])/(C[n]+A[n]*alpha[n-1])
    
    # рекуррентное вычисление неизвестных
    for i in reversed(range(0, n)):
        y[i] = alpha[i]*y[i+1]+beta[i]
    
    return y  

y01 = 1
y02 = -1
a = 1.3
b = 1.6
n = 100
result = []
xmas = []
ymas1 = []
ymas2 = []
sigma = [1, 1]
gamma = [-1, 0]
delta = [1, 3]

X = arange(a, b, 100)
Y = list(map(ans, X))
#Y1 = list(map(ans4_1, X))
#Y2 = list(map(ans4_2, X))
#make_graphic(X, Y, '-', lw=1)
#make_graphic(X, Y2, '-', 'blue', 8)
#runge2(f, a, b, y01, n, xmas, ymas1)
#y = run_through_method(a, b, p2, q2, f2, n, sigma, gamma, delta)
y = boundary(p, q, f, sigma, gamma, delta, n, a, b)
X = [a+i*((b-a)/n) for i in range(0, n+1)]
make_graphic(X, y, "-", lw=1)
#make_graphic(xmas, ymas2, "-", "purple", 4)
#plt.legend(['y точное', 'y численное'])
#plt.title('Краевая задача, n = %d' % n)
plt.show()
