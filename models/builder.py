import gc

import data_loader as dl
from . import content_based, collaborative, knowledge_graph

_prod_dept: dict = {}


def build_all() -> None:
    global _prod_dept
    _prod_dept = {int(k): int(v) for k, v in dl.prod_dept_map.items()}

    print("\n" + "=" * 65)
    print("BUILD — Content-Based")
    print("=" * 65)
    content_based.build(_prod_dept)

    print("\n" + "=" * 65)
    print("BUILD — Collaborative (SPMI)")
    print("=" * 65)
    spmi = collaborative.build(_prod_dept)

    print("\n" + "=" * 65)
    print("BUILD — Knowledge Graph (RWR)")
    print("=" * 65)
    knowledge_graph.build(spmi)

    del spmi
    gc.collect()
    print("\n[OK] All models ready.\n")
