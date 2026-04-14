from dataset import EnglishDummyDataset
from ruma_model import RUMAModel


def run_memory_update_smoke_test():
    print("\n=======================================================")
    print("   RUMA MEMORY UPDATE PATH SMOKE TEST")
    print("=======================================================\n")

    dataset = EnglishDummyDataset()
    model = RUMAModel(vocab_size=dataset.vocab_size, d_model=32, n_heads=2, num_shards=4)

    print("[1/3] Memory store before writes:")
    print(model.memory_stats())

    x_a, _ = dataset.get_data_A()
    x_b, _ = dataset.get_data_B()
    write_batch = x_a.new_empty((2, x_a.shape[1]))
    write_batch[0] = x_a[0]
    write_batch[1] = x_b[0]

    print("\n[2/3] Writing two toy sequences into routed memory shards...")
    model.update_memory(
        write_batch,
        sources=["base_phrase", "new_phrase"],
        namespaces=["toy", "toy"],
        timestamps=["t0", "t1"],
    )
    print(model.memory_stats())

    print("\n[3/3] Running a forward pass after memory insertion...")
    logits, aux = model(x_a, return_aux=True)
    print(f"Output logits shape: {list(logits.shape)}")
    print(f"Route map shape:      {list(aux['routes'].shape)}")
    print("\n[NOTE] This test only verifies that the explicit memory write/read path")
    print("       is wired into the sandbox. It does not prove quality or retention.")


if __name__ == "__main__":
    run_memory_update_smoke_test()
