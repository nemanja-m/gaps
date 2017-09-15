import numpy as np
import cv2
import bisect
from matplotlib import pyplot as plt


def possible_piece_sizes(rows, columns):
    sizes = []
    min_size = 28
    max_size = min(rows, columns) / 2
    for size in range(min_size, max_size):
        if rows % size == 0 and columns % size == 0:
            sizes.append(size)
    return sizes

def apply_threshold(channel):
    _, thresh = cv2.threshold(channel, 200, 255, cv2.THRESH_BINARY)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (5, 5), iterations=3)
    return cv2.bitwise_not(opened)

def find_size_candidates(channel):
    binary_image = apply_threshold(channel)
    rows, columns = binary_image.shape
    sizes = possible_piece_sizes(rows, columns)

    _, contours, _ = cv2.findContours(binary_image,
                                      cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)

    filtered = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if abs(h - w) < 3 and w > sizes[0] and w < columns:
            filtered.append(w)
    return filtered

def find_nearest_size(size_candidate, possible_sizes):
    index = bisect.bisect_right(possible_sizes, size_candidate)

    if index == 0:
        return possible_sizes[0]

    if index >= len(possible_sizes):
        return possible_sizes[-1]

    right_size = possible_sizes[index]
    left_size = possible_sizes[index - 1]

    if abs(size_candidate - right_size) < abs(size_candidate - left_size):
        return right_size
    else:
        return left_size

def detect_piece_size(image):
    blue, green, red = cv2.split(image)

    rows, columns = red.shape
    possible_sizes = possible_piece_sizes(rows, columns)
    print possible_sizes

    if len(possible_sizes) == 1:
        print possible_sizes[0]
        return possible_sizes[0]

    channels = [
        red,
        green,
        blue,
        cv2.add(red, green),
        cv2.add(red, blue),
        cv2.add(green, blue)
    ]

    size_candidates = []
    for channel in channels:
        candidates = find_size_candidates(channel)
        size_candidates.extend(candidates)

    sizes_map = { size: 0 for size in possible_sizes }

    for size_candidate in size_candidates:
        nearest_size = find_nearest_size(size_candidate, possible_sizes)
        sizes_map[nearest_size] += 1

    piece_size = max(sizes_map, key=sizes_map.get)
    print piece_size
    return piece_size


image = cv2.imread('out.jpg')
detect_piece_size(image.copy())
