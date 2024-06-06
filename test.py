import DE_HS_part as de
import numpy as np
import cv2

img = cv2.imread("image\lena.bmp",cv2.IMREAD_GRAYSCALE)
img_embeded=cv2.imread("embeded_image\embeded_0.png",cv2.IMREAD_GRAYSCALE)
img_recover=cv2.imread("embeded_image\\recover_0.png",cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape[:2]
matrix_h,matrix_a=de.difference_average_calculate(img,rows,cols)
matrix_h_2,matrix_a_2=de.difference_average_calculate(img_embeded,rows,cols)
de.image_check(img,img_recover,rows,cols)

embedded_bits=np.loadtxt('embeded_image\\bitstream_origin.txt')
extracted_bits=np.loadtxt('embeded_image\\message_extract.txt')
de.bitstream_check(extracted_bits,embedded_bits)

de.hist_demo_bin(matrix_h)
#de.hist_demo_bin(matrix_h_2)
#de.ist_demo_line(matrix_h)
#de.hist_demo_line(matrix_a)
# 计算可扩展位置的函数
'''
expan_pos_0,expan_pos_1,chang_pos_0,chang_pos_1,unexpan_pos=expansionable_changeable_bit_calculate(img_highpass,matrix_a,rows,cols)
print(expan_pos_0,expan_pos_1,chang_pos_0,chang_pos_1,unexpan_pos)
print(len(expan_pos_0)/2,len(expan_pos_1)/2,len(chang_pos_0)/2,len(chang_pos_1)/2,len(unexpan_pos)/2)'''
n=2000
bitstream_embeded=np.random.randint(0, 2, n)  # 嵌入的信息, 0,1 交替的序列，n为长度

# 选中函数的调用
#dict,select_delta,select_num=de.embed_pos_select(matrix_h,matrix_a,rows,cols,bitstream_embeded)
#de.img_embed(img,matrix_h,matrix_a,rows,cols,bitstream_embeded)
de.img_recovery(img_embeded,rows,cols)
#print(len(dict[0]),len(dict[1]),len(dict[-1]))
cv2.waitKey(0)
cv2.destroyAllWindows()