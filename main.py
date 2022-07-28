import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import operator


def detect_lines(image):
    # 输入原始图, 返回 lines(直线数组)
    # 灰度化.
    gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # 二值化. 九宫线条非常工整且贴近纯黑色, 是很好的目标
    ret, binar_img = cv.threshold(gray_img, 3, 255, cv.THRESH_BINARY)
    # 边缘检测. 似乎不可略
    edges_img = cv.Canny(binar_img, 50, 150, apertureSize=3)
    # 直线检测. 统计概率霍夫线变换
    lines = cv.HoughLinesP(edges_img, 1, np.pi / 180, 100, np.array([]), minLineLength=540, maxLineGap=200)
    return lines


def optimize_lines(lines):
    # 使用横/纵 list 归类 lines
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if abs(x1 - x2) < 2:
            horizontal_lines.append(line)
            # print("horizontal lines:", line)
        else:
            vertical_lines.append(line)
            # print("vertical lines:", line)

    # [array([[1077, 2127, 1077,  210]], dtype=int32), array([[ 357, 2126,  357, 1489]], dtype=int32),......]
    # lines 类型是 list, 它的元素是 array, 不知道这种混合模式怎么排序, 查查算法自己实现
    # 对于9宫格, 只需要确定3个 x 和3个 y, 即可判断出每一宫的范围
    x_list = []
    for i in range(len(horizontal_lines)):
        x1, y1, x2, y2 = horizontal_lines[i].reshape(4)
        x_list.append(x1)

    y_list = []
    for i in range(len(vertical_lines)):
        x1, y1, x2, y2 = vertical_lines[i].reshape(4)
        x_list.append(y1)

    print("x_list:", y_list)  # [1077, 357, 717, 360, 720, 0, 719, 716, 357]
    x_list.sort(key=None, reverse=False)
    print("x_list sort: ", y_list)  # [0, 357, 357, 360, 716, 717, 719, 720, 1077]
    print(len(y_list))

    print("y_list:", y_list)  # [1077, 357, 717, 360, 720, 0, 719, 716, 357]
    y_list.sort(key=None, reverse=False)
    print("y_list sort: ", y_list)  # [0, 357, 357, 360, 716, 717, 719, 720, 1077]
    print(len(y_list))

    # 5个像素内的 x 值去重复(计算均值), 由于宫格线有几个像素宽度导致识别成多条直线
    x_res = []
    y_res = []
    tmp_stack = []
    for i in range(len(x_list)):    # 注意如果 len 是 10, i 只会取到 0-9
        print(i)
        # print(x_list[j])
        # 首元素没有前一元素
        if i == 0:
            tmp_stack.append(x_list[i])
            continue

        # 如果与前一元素贴近就压入临时栈 tmp_stack, 等待计算均值
        if abs(x_list[i] - x_list[i - 1]) <= 5:
            tmp_stack.append(x_list[i])
            continue
        else:
            # 未检测到相近值, 计算临时栈之后持久化
            print("tmp_stack: ", tmp_stack)
            temp = int(sum(tmp_stack) / len(tmp_stack))
            print(temp)
            x_res.append(temp)

        if i == (len(x_list)-1):
            # 最后一个元素与前值不相近, 直接持久化
            x_res.append(x_list[i])
        else:
            # 非最后元素, 压入临时栈
            tmp_stack.clear()
            tmp_stack.append(x_list[i])
    print(x_res)

    for i in range(len(y_list)):    # 注意如果 len 是 10, i 只会取到 0-9
        print(i)
        # print(y_list[j])
        # 首元素没有前一元素
        if i == 0:
            tmp_stack.append(y_list[i])
            continue

        # 如果与前一元素贴近就压入临时栈 tmp_stack, 等待计算均值
        if abs(y_list[i] - y_list[i - 1]) <= 5:
            tmp_stack.append(y_list[i])
            continue
        else:
            # 未检测到相近值, 计算临时栈之后持久化
            print("tmp_stack: ", tmp_stack)
            temp = int(sum(tmp_stack) / len(tmp_stack))
            print(temp)
            y_res.append(temp)

        if i == (len(y_list)-1):
            # 最后一个元素与前值不相近, 直接持久化
            y_res.append(y_list[i])
        else:
            # 非最后元素, 压入临时栈
            tmp_stack.clear()
            tmp_stack.append(y_list[i])
    print(y_res)

    # merge_lines = []
    # merge_lines.extend(horizontal_lines)
    # merge_lines.extend(vertical_lines)
    # lines_reshape = []
    # for line in horizontal_lines:
    #     print(line.reshape(4))
    #     lines_reshape.append(line.reshape(4))
    #
    # lines_reshape.sort(key=lambda x: [x][0][0], )
    # print(horizontal_lines)


def drawing_lines(image, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(image, (x1, y1), (x2, y2), (255, 0, 0), 8)
        plt.imshow(image)
        plt.show()


img = cv.imread("photo.source\\3.jpg")
print(img.shape)
print()

mylines = detect_lines(img)
optimize_lines(mylines)
drawing_lines(img, mylines)
