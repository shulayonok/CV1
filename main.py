from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import configuration as config

func = lambda x, y, center, sigma: np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)) / (
            2 * np.pi * sigma ** 2)


# Инвертирование
def invert(arr):
    X, Y, U = arr.shape
    result = np.zeros((X, Y, U), dtype=int)
    for i in range(X):
        for j in range(Y):
            for k in range(U):
                result[i, j, k] = 255 - arr[i, j, k]
    return result


# ЧБ
def black_and_white(arr):
    X, Y, U = arr.shape
    result = np.zeros((X, Y), dtype=int)
    for i in range(X):
        for j in range(Y):
            result[i, j] = np.mean(arr[i, j])
    return result


# Шум
def random_noise(arr):
    X, Y, U = arr.shape
    noise = np.random.normal(config.mu, config.sigma, (X, Y, U))
    result = np.array(arr + noise, dtype=np.uint8)
    np.where(result < 255, result, 255)
    np.where(result > 0, result, 0)
    return result


# Гистограмма
def to_hist(arr):
    X, Y = arr.shape
    result = np.zeros(256)
    for i in range(X):
        for j in range(Y):
            result[arr[i, j]] += 1
    return result


# Наложение фильтра
def filter(arr, shape):
    X, Y, U = arr.shape
    center = shape // 2
    borderX, borderY, borderZ = arr.shape
    print(borderZ)
    # Добавляем рамку
    borderX += center * 2
    borderY += center * 2
    result = np.zeros((borderX, borderY, borderZ), dtype=np.uint8)
    # Внутрь помещаем изображение
    result[center:-center, center:-center] = arr
    # Генерим фильтр
    matrix = gauss(shape)
    # Накладываем фильтр
    for i in range(X):
        for j in range(Y):
            for k in range(U):
                result[i + center, j + center, k] = int(np.sum(result[i:i + shape, j:j + shape, k] * matrix))
    return result[center:-center, center:-center]


# Формирование фильтра определённой размерности
def gauss(shape):
    matrix = np.zeros((shape, shape))
    center = shape // 2
    for i in range(shape):
        for j in range(shape):
            matrix[i, j] = func(i, j, center, config.sigma)
    matrix /= np.sum(matrix)
    return matrix


def to_norm_hist(arr):
    hist, bins = np.histogram(arr, 256)
    norm = hist.cumsum()
    norm = (norm - norm[0]) * 255 // (norm[-1] - 1)
    return to_hist(norm[arr])


# 1. Считываем в массив
plt.subplot(3, 3, 1)
imageArr = np.array(Image.open("one.jpg"))
plt.imshow(imageArr)

# 2. Инвертируем
plt.subplot(3, 3, 2)
plt.imshow(invert(imageArr))

# 3. В ЧБ
plt.subplot(3, 3, 3)
bwImageArr = black_and_white(imageArr)
plt.imshow(bwImageArr, cmap='gray')

# 4. Пошумим
plt.subplot(3, 3, 4)
plt.imshow(random_noise(imageArr))

# 5. Гистограмма
plt.subplot(3, 2, 4)
plt.bar(range(256), to_hist(bwImageArr))

# 6. Гаусс
plt.subplot(3, 3, 7)
gaussImageArr = filter(imageArr, 15)
plt.imshow(gaussImageArr)

# 7. Нерезкое маскирование
plt.subplot(3, 3, 8)
plt.imshow(imageArr - gaussImageArr)

# 8. Гистограмма яркости
plt.subplot(3, 3, 9)
plt.bar(range(256), to_norm_hist(bwImageArr))

plt.show()
