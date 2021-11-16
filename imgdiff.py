import cv2


def get_difference_image(img_path1, img_path2):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    return __create_bounding_rect(img=cv2.imread(img_path2), diff=cv2.absdiff(img1, img2), alpha=0.4, difference_size_to_be_excluded=5)


def __create_bounding_rect(img, diff, line_color: tuple = None, fill_color: tuple = None, alpha: float = 0.4, difference_size_to_be_excluded=0):
    if fill_color is None:
        fill_color = (243, 166, 248)
    if line_color is None:
        line_color = (0, 0, 255)

    overlay = img.copy()
    contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > difference_size_to_be_excluded and h > difference_size_to_be_excluded:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), line_color, 2)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), fill_color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


if __name__ == '__main__':
    result_img = get_difference_image('./20_comp.png', './20_ref.png')
    cv2.imshow('result image', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
