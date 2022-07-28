import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



img = cv.imread("photo.source\\3.jpg")
print(img.shape)
print()

# cv.imshow("img", img)
# cv.waitKey(0)

gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# cv.imshow("gray", gray)
# cv.waitKey(0)

# blur = cv.GaussianBlur(gray, (5, 5), 0)
# cv.imshow("gauss blur", blur)
# cv.waitKey(0)

# edges = cv.Canny(gray, 10, 10)
# cv.imshow("edges", edges)
# cv.waitKey(0)
ret, thresh1 = cv.threshold(gray, 3, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(gray, 5, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(gray, 200, 255, cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(gray, 200, 255, cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(gray, 200, 0, cv.THRESH_TOZERO_INV)
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')  # 将图像按2x3铺开
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

# cv.imshow("binar", binar)
# cv.waitKey(0)

edges = cv.Canny(thresh2, 50, 150, apertureSize=3)
plt.imshow(edges)
plt.show()

# cv.imshow("edges", edges)
# cv.waitKey(0)

# plt.imshow(edges)
# plt.show()


lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, np.array([]), minLineLength=540, maxLineGap=200)
# lines = cv.HoughLines(edges, 1, np.pi / 180, 180)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # x1, y1, x2, y2 = line[0]
        cv.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 10)
        print(line)

# cv.imshow("line_detect_possible_demo", edges)
# cv.waitKey(0)

plt.imshow(edges)
plt.show()





# 标准霍夫线变换
def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3
    cv.imshow("edges", edges)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 80)
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
    a = np.cos(theta)  # theta是弧度
    b = np.sin(theta)
    x0 = a * rho  # 代表x = r * cos（theta）
    y0 = b * rho  # 代表y = r * sin（theta）
    x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
    y1 = int(y0 + 1000 * a)  # 计算起始起点纵坐标
    x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
    y2 = int(y0 - 1000 * a)  # 计算直线终点纵坐标 注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
    cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 点的坐标必须是元组，不能是列表。
    cv.imshow("image-lines", image)


# 统计概率霍夫线变换
def line_detect_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=5)
    for line in lines:
        x1, y1, x2, y2 = line[0]
    cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("line_detect_possible_demo", image)
