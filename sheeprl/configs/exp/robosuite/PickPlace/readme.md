## PickPlace Benchmarks

This was the most performant run for the PickPlace task with our custom scenes. It makes use of all 4 reward shaping objectives and manages an average smoothed reward of 84.62, managing to sporadically finish the task a couple times as per the tensorboard `ep_len` metric.

To guarantee replication make sure you first have the correct environment (found in `PickPlace_benchmark_env.yml`) and run with

```
python sheeprl.py exp=robosuite/PickPlace/sac_pick_up_tomato_can.yaml
```

The raw overrides from the run logs are also available in `sac_pick_up_tomato_can_raw.yaml`