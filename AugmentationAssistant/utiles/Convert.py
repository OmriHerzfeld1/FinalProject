

def seg_to_key(seg):
    """
    convert segmentations format to keypoints format
    :param seg: [x0, y0, x1, y1, ...]
    :return: format - [(x0,y0), (x1,y1), (x2,y2), ....]
    """
    keypoints: list = []
    for i in range(0, len(seg)):  # moving segmentation to key point format
        if i % 2 == 0:  # index is even  value is x
            x = seg[i]
        else:
            keypoints.append((x, seg[i]))
    return keypoints


def key_to_seg(keypoints):
    """
    convert keypoints format to segmentations format
    :param keypoints: format - [(x0,y0), (x1,y1), (x2,y2), ....]
    :return: [x0, y0, x1, y1, ...]
    """
    segmentations: list = []
    for point in keypoints:
        segmentations.append(round(point[0]))
        segmentations.append(round(point[1]))
    return segmentations