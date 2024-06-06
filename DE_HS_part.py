import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from math import pow
'''
水印嵌入流程
1.获取直方图中差值为0的数目，生成随机比特流，和图像四角的lsb组成嵌入信息
2.按照差值为0、-1、1...的顺序选择可扩展像素对，根据嵌入长度进行截取，进行信息嵌入，注意这里不要选择图像四角
3.画出扩展后的直方图，对重叠部分的阈值进行计算
4.根据阈值，对未被选择的差值进行移位操作：可以是左边加，或者是拆分左加右减
5.从图像左上角开始，按照顺时针顺序把阈值嵌入到lsb位
6.差分扩展，输出图像
'''

'''
水印提取流程
1.按照顺序读取图像四角的lsb，确定阈值
2.画出差值图像和均值图像，对阈值范围内的差值图像进行水印提取
3.截取嵌入信息的最后四位，对图像四角的lsb进行还原
4.对剩下所有像素进行直方图移位，还原原像素点
5.把被嵌入的像素点进行恢复
5.比对嵌入的信息和恢复的图像，验证其正确性
'''
# 根据差值和均值还原像素点
def recover_HL(h, l):
    if h >= 0:
        x = l + int((h + 1) / 2)
        y = l - int(h / 2)
    else:
        x = l + int((h) / 2)
        y = l - int((h - 1) / 2)
    return x, y

def hist_demo_line(img):#画带折线的直方图的函数
    cv2.imshow("img", img)
    # 创建一个画布
    fig, ax = plt.subplots()
    # 绘制直方图
    num, bins_limit, patches = ax.hist(img.ravel(), bins=512, range=(-256, 256))
    # 曲线图
    ax.plot(bins_limit[:512], num, '-',  color='red')
    #           x轴          y轴
    plt.show()

def hist_demo_bin(img):#画柱状直方图的函数
    cv2.imshow("img", img)
    '''
    # cv2的直方图函数仅支持uint8和float32格式
    hist = cv2.calcHist([img], [0], None, [256], [-256, 256])
    plt.plot(hist)
    '''
    plt.hist(img.ravel(), 512, [-256, 256])
    plt.show()
# 差值图像其实是相邻元素之间进行做差，最后得到的矩阵的大小会变成m*n/2；均值图像同理
# 本函数计算差值和均值图像，返回格式为：差值，均值=函数(图像，行数，列数)
def difference_average_calculate(img,rows,cols):
    matrix_difference = np.zeros((cols,int(rows/2)))
    matrix_average = np.zeros((cols, int(rows / 2)))
    for i in range(rows):
        for j in range(0,cols,2):
                matrix_difference[i, int(j/2)] = int(img[i][j]) - int(img[i][j + 1])
                matrix_average[i, int(j / 2)] = int((int(img[i][j]) + int(img[i][j + 1]))/2)
    return matrix_difference,matrix_average
# 计算可扩展位置、可改变位置，实际使用意义不大，因为可扩展和可改变是相对于嵌入比特流的
# 代码没问题，运行时间较长
# lena图像不存在不可扩展的位置（第一遍）
def expansionable_changeable_bit_calculate(matrix_difference,matrix_average,rows,cols):
    expan_pos_0=np.array([])# 可扩展0
    expan_pos_1 = np.array([])# 可扩展1
    chang_pos_0=np.array([])# 可改变0
    chang_pos_1 = np.array([])#可改变1
    unexpan_pos=np.array([])# 不可扩展的位置
    start_time = time.time()
    for i in range(rows):
        for j in range(int(cols/2)):
            print('正在处理像素对',i,j,i,j+1)
            # 确定当前的可扩展阈值
            threhold0fexpan=min(2*(255-matrix_average[i,j]),2*matrix_average[i,j]+1)
            # 可扩展0
            if abs(2*matrix_difference[i,j])<=threhold0fexpan:
                expan_pos_0=np.append(expan_pos_0,[i,j])
            # 可扩展1
            if abs(2*matrix_difference[i,j]+1)<=threhold0fexpan:
                expan_pos_1=np.append(expan_pos_1,[i,j])
            # 可改变0
            if abs(2 * int(matrix_difference[i,j] / 2))<=threhold0fexpan:
                chang_pos_0=np.append(chang_pos_0,[i,j])
            # 可改变1
            if abs(2 * int(matrix_difference[i,j] / 2)+1)<=threhold0fexpan:
                chang_pos_1=np.append(chang_pos_0,[i,j])
            # 不可扩展
            else:
                unexpan_pos=np.append(unexpan_pos,[i,j])
    end_time = time.time()
    print('代码执行时间:',end_time-start_time,'s')
    return expan_pos_0,expan_pos_1,chang_pos_0,chang_pos_1,unexpan_pos
# 嵌入位置选择函数,计算可扩展1的位置并进行阈值选择
def embed_pos_select(matrix_difference,matrix_average,rows,cols,bitstream):
    # 用来记录可扩展差值的字典，格式为{i:[x1,y1],[x2,y2]}
    dict = {}
    # 一个是记录在选择阈值的前提下可嵌入的长度，一个是阈值
    selected_pos_num=0;delta=0
    for i in range(-255,256):
        dict[i] = []
    for i in range(rows):
        for j in range(int(cols/2)):
            print('正在处理像素对',i,2*j,i,2*j+1)
            # 确定当前的可扩展阈值
            threhold0fexpan=min(2*(255-matrix_average[i,j]),2*matrix_average[i,j]+1)
            # 我们考虑最坏的情况，可扩展1肯定可扩展0，这里只统计可扩展1的数量
            if abs(2*matrix_difference[i,j]+1)<=threhold0fexpan:
                dict[matrix_difference[i,j]].append([i,j])
    # 确定选择像素差值范围的函数
    selected_pos_num+=len(dict[0])
    while selected_pos_num<len(bitstream):
        selected_pos_num+=len(dict[-delta])
        selected_pos_num += len(dict[delta])
        delta+=1
    if delta>7:
        print('隐藏信息太长，请重新选择！')
        delta=-999
    return dict,delta,selected_pos_num

# 水印嵌入函数
def img_embed(origin_img,matrix_difference,matrix_average,rows,cols,bitstream):
    dict, select_delta, select_num = embed_pos_select(matrix_difference, matrix_average, rows, cols, bitstream)
    # 先把原图片矩阵复制一份，然后根据所选阈值进行直方图移位
    image_embed=origin_img
    # 根据选择的值确定正数和负数各自的移动长度，也就是说我们选择是按照类似[0]、[-1,0,1]这样选择的,把最大值记为delta并嵌入
    # 扩展之后得到的范围就是[-2*delta,2delta+1]，表示了正差值和负差值的移动长度，记为positive和negative
    # 但是不可能嵌入的比特流长度刚好和选择长度相等，要进行选择；这里直接把所有像素都移出被选择的差值范围，嵌入后直接进行像素代替
    # 因为要考虑0，所以最后的移动阈值统一为2*delta+1，0相关的和负值一起处理，往x轴负方向移动
    shifting_length=2*select_delta+1
    # 直方图移位策略：针对差值为0的像素对，统一左移，对第二个像素点的值进行增加操作，例如(0,0)->(0,shifting_length)
    # 对于其他的差值，正的就减少第二个坐标的值，负的就增加第二个坐标的值，暂且不考虑像素溢出问题
    # 采用策略左移：x_=x,y_=(y+shf)%256 右移：x_=x,y_=(y-shf)%256 可以把溢出的值转为一个很大的差值 虽然会造成图像局部失真
    # 但由于差值大于120的差值基本不存在，所以基本不用担心重叠的问题
    ### 其实，在符合高斯分布的差值图像中，基本不可能出现类似这种溢出的现象
    for i in range(rows):
        for j in range(int(cols/2)):
            # 如果这一点的差值为正数，则往正向移动，即y值减小
            if matrix_difference[i,j]>0:
                image_embed[i,2*j+1]=(image_embed[i,2*j+1]-shifting_length)%256
            # 如果为负值或者是0，往负向移动,也就是y值增大
            elif matrix_difference[i,j]<=0:
                image_embed[i, 2*j+1] = (image_embed[i, 2*j+1] + shifting_length) % 256

    cv2.imwrite(f'embeded_image\shifted_delta_{0}.png'.format(select_delta),image_embed)
    # 按照顺序保存四个角的像素值，分别为左上角、右上角、右下角和左下角
    pixel_corner=[]
    pixel_corner.append(image_embed[0,0])
    pixel_corner.append(image_embed[0,cols-1])
    pixel_corner.append(image_embed[rows-1,cols-1])
    pixel_corner.append(image_embed[rows-1,0])
    # 先保存下来四个角的lsb，按照左上角、右上角、右下角和左下角的顺序，然后进行移位的四位二进制数字转换
    shf = bin(select_delta)
    shf=shf.replace('0b','')
    str_0='000'
    shf=str_0+shf
    print(shf)
    for i in range(1,5):
        print(bin(pixel_corner[i-1]))
        # 保存进bitstream
        bitstream = np.append(bitstream, int(pixel_corner[i-1] % 2))
        # 替换原先的lsb
        if int(shf[i-5]) == 1:
            pixel_corner[i-1] = pixel_corner[i-1] | 1
        elif int(shf[i-5]) == 0:
            pixel_corner[i-1] = pixel_corner[i-1] & ~ 1
        print(bin(pixel_corner[i - 1]))
    np.savetxt('embeded_image\\bitstream_origin.txt', bitstream, fmt='%d', delimiter=' ')

    # 还原到原图像上
    image_embed[0, 0]=pixel_corner[0]
    image_embed[0,cols-1]=pixel_corner[1]
    image_embed[rows-1,cols-1]=pixel_corner[2]
    image_embed[rows-1,0]=pixel_corner[3]

    select_de_array_pre=[]
    # 所有备选像素中都排除掉四角的像素值，按照光栅扫描顺序选择可扩展的差值:对dict进行重拍，最高级i从小到大，次高级j从小到大，最后截取前len(bitstream)+4
    for i in range(-select_delta, select_delta+1):
        for m in range(len(dict[i])):
            select_de_array_pre.append(dict[i][m])
    select_de_array_pre = list(filter(lambda x: x != [0, 0] and x != [rows - 1, 0] and x != [0, int(cols / 2) - 1] and x != [rows - 1,int(cols / 2) - 1],select_de_array_pre))
    # 按照光栅扫描顺序进行排序，然后截取
    select_de_array_pre = sorted(select_de_array_pre,key=lambda x:(x[0],x[1]))
    select_de_array=select_de_array_pre[0:len(bitstream)]
    # 差分扩展
    for bit in range(len(bitstream)):
        # 定位差值和均值，其中y的取值范围为0-255
        x=select_de_array[bit][0]
        y=select_de_array[bit][1]
        # 这里的差值要进行扩展，因为都是可扩展的，所以直接乘二加上第bit位；均值不变
        x_prime,y_prime=recover_HL(matrix_difference[x,y]*2+bitstream[bit],matrix_average[x,y])
        # 根据保存的坐标对，在移位后的图像上替换下来
        image_embed[x,2*y]=x_prime
        image_embed[x, 2 * y+1] = y_prime
    cv2.imwrite(f'embeded_image\embeded_{0}.png'.format(select_delta), image_embed)
    print('水印嵌入完成，保存路径为/embeded_image')

# 水印提取和图像恢复函数
def img_recovery(img_embeded,rows,cols):
    # 首先提取出四角的lsb，并转成阈值
    delta=0
    # 提取出的比特流保存
    bitstream_extract=np.array([])
    # 比特流长度
    bitstream_length = 0
    img_recover=img_embeded
    img_corners=[]
    img_corners.append(img_recover[0,0])
    img_corners.append(img_recover[0, cols-1])
    img_corners.append(img_recover[rows-1,cols-1])
    img_corners.append(img_recover[rows-1,0])
    # 计算阈值
    for i in range(4):
        delta+=(img_corners[i]%2)*pow(2,i)
    matrix_difference,matrix_average=difference_average_calculate(img_recover,rows,cols)

    # 计算正负差值的移动长度
    delta_positive=int(2*delta+1)
    delta_negative = int(2 * delta+1)
    # 按照光栅扫描顺序，按照直方图顺序从中挑选出范围内的差值存入数组；
    difference_pairs=[]
    for i in range(rows):
        for j in range(int(cols/2)):
            if matrix_difference[i,j]>=-2*delta and matrix_difference[i,j]<=2*delta+1:
                difference_pairs.append([i,j])
    '''
    # 把差值保存在字典里，方便直接进行移位
    for i in range(rows):
        for j in range(int(cols / 2)):
            if matrix_difference[i, j] in difference_array:
                bitstream_length+=1
                dict[matrix_difference[i, j]].append([i, j])'''
    # 秘密信息提取，截取后四位作为lsb并还原
    for pairs in difference_pairs:
        x, y = pairs
        # 提取嵌入比特流
        bit_extract_current = matrix_difference[x, y] % 2
        bitstream_extract = np.append(bitstream_extract, bit_extract_current)
    pixel_corner_re=[]
    # 最后四位代表lsb
    for i in range(-4,0):
        pixel_corner_re.append(bitstream_extract[i])
    # 恢复四角像素值
    for i in range(len(pixel_corner_re)):
        if pixel_corner_re[i]==1:
            img_corners[i]=img_corners[i]|1
        elif pixel_corner_re[i]==0:
            img_corners[i] = img_corners[i] &~1
    img_recover[0,0]=img_corners[0]
    img_recover[0, cols-1] = img_corners[1]
    img_recover[rows-1, cols-1] = img_corners[2]
    img_recover[rows-1, 0] = img_corners[3]

    # 直方图还原，之后把字典里的像素值还原并替代回去
    for i in range(rows):
        for j in range(int(cols / 2)):
            # 这种情况属于是差值为负，但是增加y值导致溢出，变成了一个很大的正数；在不重叠的前提下，对其进行特殊处理
            if matrix_difference[i,j]>0:
                if matrix_difference[i, j] >= 256 - 2 * delta_negative and matrix_difference[i, j] <= 256 - delta_negative:
                    img_recover[i, 2 * j + 1] = img_recover[i, 2 * j + 1] + 256 - delta_negative
                else:img_recover[i, 2 * j + 1]+=delta_positive
            # 这种情况类似上一种，属于是差值为正，但是增加y值导致溢出，变成了一个很大的负数；在不重叠的前提下，对其进行特殊处理
            elif matrix_difference[i,j] <= 0:
                if matrix_difference[i, j] >= -256 + 2 * delta_positive and matrix_difference[i, j] <= -256 + delta_positive:
                    img_recover[i, 2 * j + 1] = img_recover[i, 2 * j + 1] - 256 + delta
                else:img_recover[i, 2 * j + 1]-=delta_negative
    # 重新计算差值并还原
    for pairs in difference_pairs:
        x, y = pairs
        # 计算原差值差值并还原
        difference_current = matrix_difference[x, y] / 2
        x_re, y_re = recover_HL(difference_current, matrix_average[x, y])
        img_recover[x, 2 * y] = x_re
        img_recover[x, 2 * y + 1] = y_re
    # 输出恢复的图像
    cv2.imwrite(f'embeded_image\\recover_{0}.png'.format(delta), img_recover)
    # 保存为txt文件，并把空格作为分隔符
    np.savetxt(r'embeded_image\message_extract.txt',bitstream_extract,fmt='%d',delimiter=' ')
# 图像检验函数
def image_check(img_origin,img_recovery,rows,cols):
    flag=1
    for i in range(rows):
        if flag==1:
            for j in range(cols):
                if img_origin[i, j] != img_recovery[i, j]:
                    print(i,j,img_recovery[i, j],img_origin[i, j])
                    flag = 0
    if flag==1:
        print('图像恢复成功！')

def bitstream_check(extracted_bits,embedded_bits):
    flag=1
    if len(extracted_bits)!=len(embedded_bits):
        flag=0
        print('长度不一致，错误！')
    else:
        for num in range(len(extracted_bits)):
            if extracted_bits[num] != embedded_bits[num]:
                print("比特流提取错误！")
                flag=0
                break
    if flag==1:
        print('比特流提取成功！！！')




















