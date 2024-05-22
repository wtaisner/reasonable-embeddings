from pathlib import Path
import lzma
from joblib import Parallel, delayed

import dill

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from src.reasoner import ReasonerHead, EmbeddingLayer, train, eval_batch
from src.utils import timestr, paramcount
from src.elpp.gen import split_dataset

base_dir = Path("local/out/elpp")
base_dir.mkdir(parents=True, exist_ok=True)
ts = timestr()

reasoners_path = "reasoners/reasoners_300_concepts_5_roles_200_proofs.dill.xz"
test_reasoners_path = "reasoners/test_reasoners_300_concepts_5_roles_200_proofs.dill.xz"

seed = 2022
emb_size = 300
hidden_size = 300
epoch_count = 15
test_epoch_count = 10
batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device: {device}")

torch.set_num_threads(1)

suffix = f"emb_size_{emb_size}_hidden_size_{hidden_size}_{reasoners_path.split('/')[-1].replace('.dill.xz', '')}"

if __name__ == "__main__":

    # === experiment 1 ===
    with lzma.open(base_dir / reasoners_path, "rb") as f:
        reasoners = dill.load(f)

    artifacts = {}

    for complexity_threshold in range(2, 21):
        print("Complexity threshold", complexity_threshold)

        training, validation, test = split_dataset(
            reasoners,
            np.random.default_rng(seed=0xBEEF),
            complexity_threshold=complexity_threshold,
        )

        torch.manual_seed(seed)
        trained_reasoner = ReasonerHead(emb_size=emb_size, hidden_size=hidden_size).to(device)
        encoders = [
            EmbeddingLayer(
                emb_size=emb_size,
                n_concepts=reasoner.n_concepts,
                n_roles=reasoner.n_roles,
            ).to(device)
            for reasoner in reasoners
        ]

        print(f"created reasoner with {paramcount(trained_reasoner)} parameters")
        print(
            f"created {len(encoders)} encoders with {paramcount(encoders[0])} parameters each"
        )

        train_logger = train(
            training,
            validation,
            trained_reasoner,
            encoders,
            epoch_count=epoch_count,
            batch_size=batch_size,
            device=device,
        )

        artifacts[complexity_threshold] = {
            "reasoner": trained_reasoner,
            "encoders": encoders,
            "training": training,
            "validation": validation,
            "test": test,
        }

    tmp = {
        key: {
            "reasoner": value["reasoner"].state_dict(),
            "encoders": [e.state_dict() for e in value["encoders"]],
            "training": value["training"],
            "validation": value["validation"],
            "test": value["test"],
        }
        for key, value in artifacts.items()
    }

    with lzma.open(base_dir / f"exp1_{suffix}.dill.xz", "wb") as f:
        dill.dump(tmp, f)

    rows = []

    for complexity_threshold, components in tqdm(artifacts.items()):
        with torch.no_grad():
            idx_te, X_te, y_te = components["test"]
            _, _, Y_te_good = eval_batch(
                components["reasoner"], components["encoders"], X_te, y_te, idx_te
            )
        for i, idx in enumerate(idx_te):
            axiom = X_te[i]
            expected = y_te[i]
            predicted = Y_te_good[i]
            complexity = len(reasoners[idx].decode_shortest_proof(axiom[1], axiom[2]))
            rows.append([complexity_threshold, idx, complexity, axiom, expected, int(predicted >= 0.5), predicted])
    df = pd.DataFrame(
            rows,
            columns=[
                "Complexity threshold",
                "KB",
                "Complexity",
                "Axiom",
                "Expected",
                "Predicted",
                "Raw predicted",
            ],
        )
    df.to_feather(base_dir / f"exp1_{suffix}.feather")
    
    # === experiment 3 ===
    with lzma.open(base_dir / reasoners_path, 'rb') as f:
        reasoners = dill.load(f)
    
    with lzma.open(base_dir / f'exp1_{suffix}.dill.xz', 'rb') as f:
        artifacts = dill.load(f)

    for key, components in artifacts.items():
        neural_reasoner = ReasonerHead(emb_size=emb_size, hidden_size=hidden_size)
        neural_reasoner.load_state_dict(components['reasoner'])
        components['reasoner'] = neural_reasoner
        encoders = [EmbeddingLayer(emb_size=emb_size, n_concepts=reasoner.n_concepts, n_roles=reasoner.n_roles) for reasoner
                    in
                    reasoners]
        for sd, e in zip(components['encoders'], encoders):
            e.load_state_dict(sd)
        components['encoders'] = encoders
    with lzma.open(base_dir / test_reasoners_path, 'rb') as f:
        test_reasoners = dill.load(f)

    splits = {complexity_threshold_k: split_dataset(test_reasoners, np.random.default_rng(seed=0xbeef),
                                                complexity_threshold=complexity_threshold_k) for complexity_threshold_k
          in range(2, 21)}


    def train_helper(complexity_threshold_j, complexity_threshold_k):
        neural_reasoner = artifacts[complexity_threshold_j]["reasoner"]
        training, validation, test = splits[complexity_threshold_k]
        torch.manual_seed(seed)
        my_encoders = [EmbeddingLayer(emb_size=emb_size, n_concepts=reasoner.n_concepts, n_roles=reasoner.n_roles).to(device) for
                    reasoner in test_reasoners]
        
        train_logger = train(training, validation, neural_reasoner.to(device), my_encoders, epoch_count=epoch_count,
                            batch_size=batch_size,
                            freeze_reasoner=True, validate=False)

        with torch.no_grad():
            idx_te, X_te, y_te = test
            _, _, Y_te_good = eval_batch(neural_reasoner, my_encoders, X_te, y_te, idx_te)

        rows = []
        for i, idx in enumerate(idx_te):
            axiom = X_te[i]
            expected = y_te[i]
            predicted = Y_te_good[i]
            complexity = len(test_reasoners[idx].decode_shortest_proof(axiom[1], axiom[2]))
            rows.append([complexity_threshold_j, complexity_threshold_k, idx, complexity, axiom, expected, int(predicted >= .5), predicted])
        print(f"({complexity_threshold_j}, {complexity_threshold_k}) completed")

        return complexity_threshold_j, complexity_threshold_k, my_encoders, rows

    results = Parallel(n_jobs=-1)(
        delayed(train_helper)(complexity_threshold_j, complexity_threshold_k) for complexity_threshold_j in range(2, 21) for
        complexity_threshold_k in range(2, 21))
    
    encoders = {}
    rows = []

    for j, k, jk_encoders, some_rows in results:
        encoders[(j, k)] = jk_encoders
        rows += some_rows
    
    tmp = {
        'splits': splits,
        'encoders': {key: [e.state_dict() for e in encs] for key, encs in encoders.items()}
    }

    with lzma.open(base_dir / f'exp3_{suffix}.dill.xz', 'wb') as f:
        dill.dump(tmp, f)
    
    df = pd.DataFrame(rows, columns=["Complexity threshold j","Complexity threshold k", "KB", "Complexity", "Axiom", "Expected", "Predicted",
                                 "Raw predicted"])
    df.to_feather(base_dir / f'exp3_{suffix}.feather')
        