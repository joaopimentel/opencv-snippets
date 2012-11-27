import cv2


def search_contours(contours, h, current, criteria):
    """
    Receives hierarchy and contour list, obtained with cv2.findContours.
    Traverses recursively the hierarchy tree depth-first, goes back if current
        contour is good, according to criteria(contours, h, current).
    Returns list with indices in contours for good contours.

    Hierarchy is a one-item list, so always prepend h[0]
    Each element of h[0] is [a, b, c, d], such that:
    # a - next sibling contour
    # b - previous sibling contour
    # c - first child
    # d - parent contour
    """
    good_contours = []
    # check if current contour is good:
    if criteria(contours, h, current):
        good_contours.append(current)
    # else, check if there is a contour inside
    elif h[0][current][2] != -1:
        # there are contours inside
        good_contours += search_contours(contours, h, h[0][current][2],
                                         criteria)
    # go to next sibling (if there is one)
    if h[0][current][0] != -1:
        good_contours += search_contours(contours, h, h[0][current][0],
                                         criteria)
    # return list of good contours, including current if good
    return good_contours


if __name__ == '__main__':

    # Define a criteria function for search_contours.
    # In this example, the contour should be contained in a rectangle with area
    # between 33000 and 45000 and an aspect ratio between 4 and 3.3
    def check_area_and_aspectratio(contours, h, index):
        rect = cv2.minAreaRect(contours[index])
        # check if this contour is good: area and aspect ratio -wise
        if (45000 > (rect[1][0] * rect[1][1]) > 33000 and
            4 > max(rect[1]) / min(rect[1]) > 3.3):
            return True
        return False

    impath = '/path/to/binary/image/'
    img = cv2.imread(impath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(img,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    good_contours = search_contours(contours, hierarchy, 0,
                                    check_area_and_aspectratio)

    for contour_idx in good_contours:
        color = (0, 0, 255)
        cv2.drawContours(img_color, contours, contour_idx, color,
                         thickness=2, hierarchy=hierarchy, maxLevel=0)

    cv2.imshow('contours', img_color)
    cv2.waitKey()
