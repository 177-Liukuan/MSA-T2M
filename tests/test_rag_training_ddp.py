import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from models.rag_training import RAGTwoForwardLoss
from tests.test_rag_training import FakeRAGModel, make_batch


def _flatten_parameters(model):
    return torch.cat(
        [parameter.detach().reshape(-1).cpu() for parameter in model.parameters()]
    )


def _ddp_worker(rank, world_size, init_file, output_directory):
    dist.init_process_group(
        backend="gloo",
        init_method="file://{}".format(init_file),
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(31)
        training_module = RAGTwoForwardLoss(FakeRAGModel())
        ddp_model = DistributedDataParallel(training_module)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.05)
        batch = make_batch()
        batch["latents"] = batch["latents"] + rank * 0.25
        batch["text_emb"] = batch["text_emb"] - rank * 0.1

        optimizer.zero_grad()
        loss = ddp_model(**batch)
        loss.backward()
        optimizer.step()
        torch.save(
            _flatten_parameters(ddp_model.module),
            Path(output_directory) / "rank-{}.pt".format(rank),
        )
    finally:
        dist.destroy_process_group()


class RAGTrainingDDPTest(unittest.TestCase):
    def test_wrapped_forward_synchronizes_rank_local_gradients(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            init_file = root / "distributed-init"
            mp.spawn(
                _ddp_worker,
                args=(2, str(init_file), str(root)),
                nprocs=2,
                join=True,
            )
            rank_zero = torch.load(
                root / "rank-0.pt", map_location="cpu", weights_only=True
            )
            rank_one = torch.load(
                root / "rank-1.pt", map_location="cpu", weights_only=True
            )
            torch.testing.assert_close(rank_zero, rank_one, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
