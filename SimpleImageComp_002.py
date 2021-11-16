import numpy as np
from PIL import Image
import sys

# 画像の読み込み
image1 = Image.open("./20_comp.png")
image2 = Image.open("./20_ref.png")

# RGB画像に変換
image1 = image1.convert("RGB")
image2 = image2.convert("RGB")

# NumPy配列へ変換
im1_u8 = np.array(image1)
im2_u8 = np.array(image2)

# サイズや色数が違うならエラー
if im1_u8.shape != im2_u8.shape:
    print("サイズが違います")
    sys.exit()

# 負の値も扱えるようにnp.int16に変換
im1_i16 = im1_u8.astype(np.int16)
im2_i16 = im2_u8.astype(np.int16)

# 差分配列作成
diff_i16 = im1_i16 - im2_i16

'''ここから作成する画像によって異なる処理'''

# 差分の絶対値が0以外の輝度値を255に変換
diff_bool = diff_i16 != 0

# 色ごとに分離した配列を参照
r_bool = diff_bool[:,:,0]
g_bool = diff_bool[:,:,1]
b_bool = diff_bool[:,:,2]

# 輝度値に差がある画素の輝度値を255とするグレースケール画像の配列作成
mask_bool = r_bool | g_bool | b_bool
mask_u8 = mask_bool.astype(np.uint8) * 255

# PIL画像に変換
mask_img = Image.fromarray(mask_u8)

# 全ての画素の色を(0,255,0)とした配列作成
green_u8 = np.zeros(im1_u8.shape, np.uint8)
green_u8[:,:,1] = 255

# １つ目の画像に色を混ぜる
blend_u8 = im1_u8 * 0.75 + green_u8 * 0.25

# 画像に変換
blend_img = Image.fromarray(blend_u8.astype(np.uint8))

# １つ目の画像に貼り付け
diff_img = image1.copy()
diff_img.paste(im=blend_img, mask=mask_img)

'''ここまで作成する画像によって異なる処理'''

# 画像表示
diff_img.show()