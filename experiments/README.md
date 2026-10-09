# Extra policies

Two more ways to train on the same replays. They are not part of the main results; their policies are held-out
policies that `scripts/world_model_eval.sh` scores inside the world model (experiment 4).

| Script | What it trains | Policies |
|---|---|---|
| `more_steps.sh` | policies A and B trained 5000 more steps on their own replays | `more_base`, `shift_more` |
| `mixed_data.sh` | the moved-can and unmoved-can replays together; with the can moved it scores below the moved-can replays alone | `phone_mix_base` |
