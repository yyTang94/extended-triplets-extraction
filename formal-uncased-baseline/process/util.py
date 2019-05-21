import sys
from typing import List, Any, Callable


def find_nearest_pair(al: List[Any], am: List[bool],
                      bl: List[Any], bm: List[bool]):

    shortest_dist = sys.maxsize
    hit_a_ix = -1
    hit_b_ix = -1

    for a_ix, (a, a_mask) in enumerate(zip(al, am)):
        if a_mask is True:
            b_ix, cur_dist = find_nearest_element(a, bl, bm)

            if b_ix != -1 and cur_dist < shortest_dist:
                hit_a_ix = a_ix
                hit_b_ix = b_ix

                # update shortest distance
                shortest_dist = cur_dist

    return hit_a_ix, hit_b_ix, shortest_dist


def find_nearest_element(a: Any, bl: List[Any], bm: List[bool]):

    shortest_dist = sys.maxsize
    hit_ix = -1

    for ix, (b, mask) in enumerate(zip(bl, bm)):

        if (mask is True) and (max(a[0], b[0]) >= min(a[0] + a[1], b[0] + b[1])):
            cur_dist = abs(b[0] - a[0])

            if cur_dist < shortest_dist:
                hit_ix = ix

                # update shortest distance
                shortest_dist = cur_dist

    return hit_ix, shortest_dist
