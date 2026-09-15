import itertools
import numpy as np
from src.boolean_core import circuit_schema, input_tables, make_tasks
from src.tree_improver import develop_tree, TREE_TAGS
from src.adapter_diagnostic import develop_adapter


def test_identity_adapter_matches_original_and_fixed_is_permutation_invariant():
    for seed in range(6):
        rng = np.random.default_rng(92500+seed)
        p, y = make_tasks(rng, 3, 6, 'mixed')
        c, tags = circuit_schema(6); inp = input_tables()
        ut, um = rng.random((3, 8, 2)), rng.random((32, 2))
        ur = rng.random((32, 10))
        fixed_reference = None
        for recursive, regime in [(True, 0), (False, 1)]:
            a = develop_adapter(recursive, TREE_TAGS, p,y,c,tags,inp,ut,um,8)
            b = develop_tree(regime,p,y,c,tags,inp,ut,um,ur,np.array([32]),8)
            np.testing.assert_array_equal(a[0], b[0][0])
            assert a[1] == b[1][0] and a[2] == b[2][0]
            if not recursive:
                fixed_reference = a
        for permutation in itertools.permutations(range(3)):
            meta_tags = np.array(permutation)[TREE_TAGS]
            x = develop_adapter(False,meta_tags,p,y,c,tags,inp,ut,um,8)
            np.testing.assert_array_equal(x[0], fixed_reference[0])
            assert x[1:] == fixed_reference[1:]
