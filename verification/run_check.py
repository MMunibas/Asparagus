import asparagus

import numpy as np
import torch

from asparagus import utils

from ase import Atoms

# Check None
print("\nCheck None")
xtrue = [
    None, np.array(None)
    ]
xfalse = [
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", 
    np.array(True), np.array([None]), np.array([False]),
    [1], [1.1, 1e3], np.array([2]), torch.tensor([3]), 
    np.array([2.5, 2]), torch.tensor([3.4, 4]),
    ]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_None(xi))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_None(xi))
print("Must be False: ", np.any(rfalse))




# Check string
print("\nCheck string")
xtrue = [
    "a", np.array("a"),
    ]
xfalse = [
    None, np.array(None), 1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False,  
    np.array(True), np.array([None]), np.array([False]),
    [1], [1.1, 1e3], np.array([2]), torch.tensor([3]), 
    np.array([2.5, 2]), torch.tensor([3.4, 4]),
    ]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_string(xi))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_string(xi))
print("Must be False: ", np.any(rfalse))



# Check bool
print("\nCheck bool")
xtrue = [
    True, False,
    np.array(True), np.array(False),
    torch.tensor(True), torch.tensor(False)
    ]
xfalse = [
    "a", np.array("a"),
    None, np.array(None), 1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1,
    np.array([None]), np.array([False]), None, np.nan, np.array(None),
    [1], [1.1, 1e3], np.array([2]), torch.tensor([3]), 
    np.array([2.5, 2]), torch.tensor([3.4, 4]),
    np.array([None]), np.array([True, True]),
    ]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_bool(xi))
    #print(xi, utils.is_bool(xi))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_bool(xi))
    #print(xi, utils.is_bool(xi))
print("Must be False: ", np.any(rfalse))



# Check numeric
print("\nCheck numeric")
xtrue = [
    1, 1.1, 1e3, np.array(2), torch.tensor(3), 
    np.array(2.5), torch.tensor(3.4)]
xfalse = [
    True, False, "a", None, np.array(None), np.array(True), np.array([None]), np.array([False]),
    [1], [1.1, 1e3], np.array([2]), torch.tensor([3]), 
    np.array([2.5, 2]), torch.tensor([3.4, 4]),]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_numeric(xi))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_numeric(xi))
print("Must be False: ", np.any(rfalse))



# Check integer
print("\nCheck integer")
xtrue = [
    1, -1_000, np.array(2), torch.tensor(3)]
xfalse = [
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True), np.array([None]), np.array([False]),
    [1], [1.1, 1e3], np.array([2]), torch.tensor([3]), 
    np.array([2.5, 2]), torch.tensor([3.4, 4]),]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_integer(xi))
    #print(xi, utils.is_integer(xi))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_integer(xi))
    #print(xi, utils.is_integer(xi))
print("Must be False: ", np.any(rfalse))



# Check callable
print("\nCheck callable")
xtrue = [
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units]
xfalse = [
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True), np.array([None]), np.array([False]),
    [1], [1.1, 1e3], np.array([2]), torch.tensor([3]), 
    np.array([2.5, 2]), torch.tensor([3.4, 4]),]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_callable(xi))
    #print(xi, utils.is_callable(xi))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_callable(xi))
    #print(xi, utils.is_callable(xi))
print("Must be False: ", np.any(rfalse))



# Check dictionary
print("\nCheck dictionary")
xtrue = [
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]}]
xfalse = [
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True), np.array([None]), np.array([False]),
    [1], [1.1, 1e3], np.array([2]), torch.tensor([3]), 
    np.array([2.5, 2]), torch.tensor([3.4, 4]),]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_dictionary(xi))
    #print(xi, utils.is_dictionary(xi))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_dictionary(xi))
    #print(xi, utils.is_dictionary(xi))
print("Must be False: ", np.any(rfalse))



# Check is array like
print("\nCheck array like")
xtrue = [
    [], np.array([]), torch.tensor([]),
    np.array([None]), np.array([False]),
    [1], [1.1, 1e3], np.array([2]), torch.tensor([3]), 
    np.array([2.5, 2]), torch.tensor([3.4, 4])]
xfalse = [
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_array_like(xi))
    #print(xi, utils.is_array_like(xi))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_array_like(xi))
    #print(xi, utils.is_array_like(xi))
print("Must be False: ", np.any(rfalse))


# Check is numeric array
print("\nCheck numeric array")
xtrue = [
    [], np.array([]), torch.tensor([]),
    [1], np.array([2]), torch.tensor([3]), 
    np.array([2, 2]), torch.tensor([3, 4]),
    np.array([2., 2.]), torch.tensor([3., 4.]),
    [1.1, 1e3]]
xtrue_inhom = [
    [], np.array([]), torch.tensor([]),
    [1], np.array([2]), torch.tensor([3]), 
    np.array([2, 2]), torch.tensor([3, 4]),
    [1.1, 1e3], [[1], [2, 3]], [[1.4], [2.1, 3.]]]
xfalse = [
    [None], [None, None], [[None, None], [None, None]], np.array([None]),
    [[None]], [None, None], [[None, None], [None]],
    [False], [False, False], [[False, False], [False, False]],
    [[False, False], [False]],
    [[1], [2, 3]], [[1.4], [2.1, 3.]], np.array([None]), np.array([False]),
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
xfalse_inhom = [
    [None], [None, None], [[None, None], [None, None]], np.array([None]),
    [[None]], [None, None], [[None, None], [None]],
    [False], [False, False], [[False, False], [False, False]],
    [[False, False], [False]],
    np.array([None]), np.array([False]),
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_numeric_array(xi))
    #print(xi, utils.is_numeric_array(xi))
for xi in xtrue_inhom:
    rtrue.append(utils.is_numeric_array(xi, inhomogeneity=True))
    #print(xi, utils.is_numeric_array(xi, inhomogeneity=True))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_numeric_array(xi))
    #print(xi, utils.is_numeric_array(xi))
for xi in xfalse_inhom:
    rfalse.append(utils.is_numeric_array(xi, inhomogeneity=True))
    #print(xi, utils.is_numeric_array(xi, inhomogeneity=True))
print("Must be False: ", np.any(rfalse))


# Check is integer array
print("\nCheck integer array")
xtrue = [
    [], np.array([]), torch.tensor([]),
    [1], np.array([2]), torch.tensor([3]), 
    np.array([2, 2]), torch.tensor([3, 4]),
    ]
xtrue_inhom = [
    [], np.array([]), torch.tensor([]),
    [1], np.array([2]), torch.tensor([3]), 
    np.array([2, 2]), torch.tensor([3, 4]),
    [[1], [2, 3]]]
xfalse = [
    [None], [None, None], [[None, None], [None, None]], np.array([None]),
    [[None]], [None, None], [[None, None], [None]],
    [False], [False, False], [[False, False], [False, False]],
    [[False, False], [False]],
    [1.1, 1e3], np.array([2., 2.]), torch.tensor([3., 4.]),
    [[1], [2, 3]], [[1.4], [2.1, 3.]], np.array([None]), np.array([False]),
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3), ["a", "b"],
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
xfalse_inhom = [
    [None], [None, None], [[None, None], [None, None]], np.array([None]),
    [[None]], [None, None], [[None, None], [None]],
    [False], [False, False], [[False, False], [False, False]],
    [[False, False], [False]],
    [1.1, 1e3], [[1.4], [2.1, 3.]], np.array([None]), np.array([False]),
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3), ["a", "b"],
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_integer_array(xi))
    #print(xi, utils.is_integer_array(xi))
for xi in xtrue_inhom:
    rtrue.append(utils.is_integer_array(xi, inhomogeneity=True))
    #print(xi, utils.is_integer_array(xi, inhomogeneity=True))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_integer_array(xi))
    #print(xi, utils.is_integer_array(xi))
for xi in xfalse_inhom:
    rfalse.append(utils.is_integer_array(xi, inhomogeneity=True))
    #print(xi, utils.is_integer_array(xi, inhomogeneity=True))
print("Must be False: ", np.any(rfalse))



# Check is string array
print("\nCheck string array")
xtrue = [
    [], np.array([]), torch.tensor([]),
    ["a", "b"], ["a", "adasdb"], 
    np.array(["a", "b"]), np.array(["a", "bq2321"]),
    np.array([["a", "b"], ["a", "b"]]),
    ]
xtrue_inhom = [
    [], np.array([]), torch.tensor([]),
    ["a", "b"], ["a", "adasdb"],
    [["a", "b"], [["a", "b"]]], ["a", "adasdb", ["a", "b"]], 
    np.array(["a", "b"]), np.array(["a", "bq2321"]),
    np.array([["a", "b"], ["a", "b"]]),
    ]
xfalse = [
    [None], [None, None], [[None, None], [None, None]], np.array([None]),
    [[None]], [None, None], [[None, None], [None]],
    [False], [False, False], [[False, False], [False, False]],
    [[False, False], [False]],
    [1], np.array([2]), torch.tensor([3]), 
    np.array([2, 2]), torch.tensor([3, 4]),
    [1.1, 1e3], np.array([2., 2.]), torch.tensor([3., 4.]),
    [[1], [2, 3]], [[1.4], [2.1, 3.]], np.array([None]), np.array([False]),
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
xfalse_inhom = [
    [None], [None, None], [[None, None], [None, None]], np.array([None]),
    [[None]], [None, None], [[None, None], [None]],
    [False], [False, False], [[False, False], [False, False]],
    [[False, False], [False]],
    [1], np.array([2]), torch.tensor([3]), 
    np.array([2, 2]), torch.tensor([3, 4]),
    [[1], [2, 3]],
    [1.1, 1e3], [[1.4], [2.1, 3.]], np.array([None]), np.array([False]),
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_string_array(xi))
    #print(xi, utils.is_string_array(xi))
for xi in xtrue_inhom:
    rtrue.append(utils.is_string_array(xi, inhomogeneity=True))
    #print(xi, utils.is_string_array(xi, inhomogeneity=True))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_string_array(xi))
    #print(xi, utils.is_string_array(xi))
for xi in xfalse_inhom:
    rfalse.append(utils.is_string_array(xi, inhomogeneity=True))
    #print(xi, utils.is_string_array(xi, inhomogeneity=True))
print("Must be False: ", np.any(rfalse))



# Check is bool array
print("\nCheck bool array")
xtrue = [
    [], np.array([]), torch.tensor([]),
    [False], [False, False], [[False, False], [False, False]],
    np.array([False]), np.array([False, False]),
    np.array([[False, False], [False, False]]),
    torch.tensor([False]), torch.tensor([False, False]),
    torch.tensor([[False, False], [False, False]])
    ]
xtrue_inhom = [
    [], np.array([]), torch.tensor([]),
    [False], [False, False], [[False, False], [False, False]],
    [[False, False], [False]],
    np.array([False]), np.array([False, False]),
    np.array([[False, False], [False, False]]),
    torch.tensor([False]), torch.tensor([False, False]),
    torch.tensor([[False, False], [False, False]])
    ]
xfalse = [
    [None], [None, None], [[None, None], [None, None]], np.array([None]),
    [[None]], [None, None], [[None, None], [None]],
    ["a", "b"], ["a", "adasdb"], 
    np.array(["a", "b"]), np.array(["a", "bq2321"]),
    np.array([["a", "b"], ["a", "b"]]),
    [1], np.array([2]), torch.tensor([3]), 
    np.array([2, 2]), torch.tensor([3, 4]),
    [1.1, 1e3], np.array([2., 2.]), torch.tensor([3., 4.]),
    [[1], [2, 3]], [[1.4], [2.1, 3.]], np.array([None]),
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
xfalse_inhom = [
    [None], [None, None], [[None, None], [None, None]], np.array([None]),
    [[None]], [None, None], [[None, None], [None]],
    ["a", "b"], ["a", "adasdb"],
    [["a", "b"], [["a", "b"]]], ["a", "adasdb", ["a", "b"]], 
    np.array(["a", "b"]), np.array(["a", "bq2321"]),
    np.array([["a", "b"], ["a", "b"]]),
    [1], np.array([2]), torch.tensor([3]), 
    np.array([2, 2]), torch.tensor([3, 4]),
    [[1], [2, 3]],
    [1.1, 1e3], [[1.4], [2.1, 3.]], np.array([None]),
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_bool_array(xi))
    #print(xi, utils.is_bool_array(xi))
for xi in xtrue_inhom:
    rtrue.append(utils.is_bool_array(xi, inhomogeneity=True))
    #print(xi, utils.is_bool_array(xi, inhomogeneity=True))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_bool_array(xi))
    #print(xi, utils.is_bool_array(xi))
for xi in xfalse_inhom:
    rfalse.append(utils.is_bool_array(xi, inhomogeneity=True))
    #print(xi, utils.is_bool_array(xi, inhomogeneity=True))
print("Must be False: ", np.any(rfalse))



# Check is None array
print("\nCheck None array")
xtrue = [
    [], np.array([]), torch.tensor([]),
    [None], [None, None], [[None, None], [None, None]], np.array([None]),
    ]
xtrue_inhom = [
    [], np.array([]), torch.tensor([]),
    [None], [None, None], [[None, None], [None, None]], np.array([None]),
    [[None]], [None, None], [[None, None], [None]],
    ]
xfalse = [
    [False], [False, False], [[False, False], [False, False]],
    np.array([False]), np.array([False, False]),
    np.array([[False, False], [False, False]]),
    torch.tensor([False]), torch.tensor([False, False]),
    torch.tensor([[False, False], [False, False]]),
    ["a", "b"], ["a", "adasdb"], 
    np.array(["a", "b"]), np.array(["a", "bq2321"]),
    np.array([["a", "b"], ["a", "b"]]),
    [1], np.array([2]), torch.tensor([3]), 
    np.array([2, 2]), torch.tensor([3, 4]),
    [1.1, 1e3], np.array([2., 2.]), torch.tensor([3., 4.]),
    [[1], [2, 3]], [[1.4], [2.1, 3.]],
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
xfalse_inhom = [
    [False], [False, False], [[False, False], [False, False]],
    [[False, False], [False]],
    np.array([False]), np.array([False, False]),
    np.array([[False, False], [False, False]]),
    torch.tensor([False]), torch.tensor([False, False]),
    torch.tensor([[False, False], [False, False]]),
    ["a", "b"], ["a", "adasdb"],
    [["a", "b"], [["a", "b"]]], ["a", "adasdb", ["a", "b"]], 
    np.array(["a", "b"]), np.array(["a", "bq2321"]),
    np.array([["a", "b"], ["a", "b"]]),
    [1], np.array([2]), torch.tensor([3]), 
    np.array([2, 2]), torch.tensor([3, 4]),
    [[1], [2, 3]],
    [1.1, 1e3], [[1.4], [2.1, 3.]],
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_None_array(xi))
    #print(xi, utils.is_None_array(xi))
for xi in xtrue_inhom:
    rtrue.append(utils.is_None_array(xi, inhomogeneity=True))
    #print(xi, utils.is_None_array(xi, inhomogeneity=True))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_None_array(xi))
    #print(xi, utils.is_None_array(xi))
for xi in xfalse_inhom:
    rfalse.append(utils.is_None_array(xi, inhomogeneity=True))
    #print(xi, utils.is_None_array(xi, inhomogeneity=True))
print("Must be False: ", np.any(rfalse))




# Check is ASE Atoms array
print("\nCheck ASE Atoms array")
xtrue = [
    [], np.array([]), torch.tensor([]),
    [
        Atoms("H2", positions=np.arange(6).reshape(2, 3)),
        Atoms("OH2", positions=np.arange(9).reshape(3, 3))
        ]    
    ]
xtrue_inhom = [
    [], np.array([]), torch.tensor([]),
    [
        [
            Atoms("H2", positions=np.arange(6).reshape(2, 3)),
            Atoms("OH2", positions=np.arange(9).reshape(3, 3))
            ],
        Atoms("H2", positions=np.arange(6).reshape(2, 3)),
        ]
    ]
xfalse = [
    [
        [
            Atoms("H2", positions=np.arange(6).reshape(2, 3)),
            Atoms("OH2", positions=np.arange(9).reshape(3, 3))
            ],
        Atoms("H2", positions=np.arange(6).reshape(2, 3)),
    ],
    [None], [None, None], [[None, None], [None, None]], np.array([None]),
    [False], [False, False], [[False, False], [False, False]],
    np.array([False]), np.array([False, False]),
    np.array([[False, False], [False, False]]),
    torch.tensor([False]), torch.tensor([False, False]),
    torch.tensor([[False, False], [False, False]]),
    ["a", "b"], ["a", "adasdb"], 
    np.array(["a", "b"]), np.array(["a", "bq2321"]),
    np.array([["a", "b"], ["a", "b"]]),
    [1], np.array([2]), torch.tensor([3]), 
    np.array([2, 2]), torch.tensor([3, 4]),
    [1.1, 1e3], np.array([2., 2.]), torch.tensor([3., 4.]),
    [[1], [2, 3]], [[1.4], [2.1, 3.]],
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
xfalse_inhom = [
    [None], [None, None], [[None, None], [None, None]], np.array([None]),
    [[None]], [None, None], [[None, None], [None]],
    [False], [False, False], [[False, False], [False, False]],
    [[False, False], [False]],
    np.array([False]), np.array([False, False]),
    np.array([[False, False], [False, False]]),
    torch.tensor([False]), torch.tensor([False, False]),
    torch.tensor([[False, False], [False, False]]),
    ["a", "b"], ["a", "adasdb"],
    [["a", "b"], [["a", "b"]]], ["a", "adasdb", ["a", "b"]], 
    np.array(["a", "b"]), np.array(["a", "bq2321"]),
    np.array([["a", "b"], ["a", "b"]]),
    [1], np.array([2]), torch.tensor([3]), 
    np.array([2, 2]), torch.tensor([3, 4]),
    [[1], [2, 3]],
    [1.1, 1e3], [[1.4], [2.1, 3.]],
    {}, {"a": 12}, {"a": 12, "b": np.nan}, {"a": 12, "b": [2.,42.]},
    np.isnan, torch.tensor, utils.is_integer, asparagus.utils.check_units,
    1, -1_000, np.array(2), torch.tensor(3),
    np.array(2.5), torch.tensor(3.4), 1.1, True, False, "a", None, 
    np.array(None), np.array(True)]
rtrue = []
rfalse = []
for xi in xtrue:
    rtrue.append(utils.is_ase_atoms_array(xi))
    #print(xi, utils.is_ase_atoms_array(xi))
for xi in xtrue_inhom:
    rtrue.append(utils.is_ase_atoms_array(xi, inhomogeneity=True))
    #print(xi, utils.is_ase_atoms_array(xi, inhomogeneity=True))
print("Must be True: ", np.all(rtrue))
for xi in xfalse:
    rfalse.append(utils.is_ase_atoms_array(xi))
    #print(xi, utils.is_ase_atoms_array(xi))
for xi in xfalse_inhom:
    rfalse.append(utils.is_ase_atoms_array(xi, inhomogeneity=True))
    #print(xi, utils.is_ase_atoms_array(xi, inhomogeneity=True))
print("Must be False: ", np.any(rfalse))
