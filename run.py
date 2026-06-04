import numpy as np
np.random.seed(42)

import data_loader as dl
import dept_direction as dd
import models
from evaluate import run_comparison

dl.load_all()
dd.build()
models.build_all()

run_comparison(
    models_dict={
        "Content-Based":        models.content_based_rec,
        "Collaborative (SPMI)": models.collab_rec,
        "Knowledge Graph":      models.kg_rec,
        "Hybrid":               models.hybrid_rec,
    },
    ks=(10, 50, 100),
)
