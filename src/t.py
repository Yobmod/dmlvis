from typing import List, Tuple, Set, Sequence, MutableSequence, Collection, Dict


l: List[str] = ["aaa", "bbb", "ccc"]
t: Tuple[str, str, str] = ("aaa", "bbb", "ccc")

s: Set[str] = {"aaa", "bbb", "ccc"}

seq: Sequence[str] = ["aaa", "bbb", "ccc"]
coll: Collection[str] = ["aaa", "bbb", "ccc"]
dic: Dict[str, bool] = {"aaa": True, "bbb": False, "ccc": True}

for x in "str":
    print(x)

for x in l:
    print(x)

for x in t:
    print(x)

for x in s:
    print(x)

for x in seq:
    print(x)

for x in dic:
    print(x)

for x in seq:
    print(x)
