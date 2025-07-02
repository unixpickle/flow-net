import os
import torch
import torch.nn as nn

from flow_net_dev.layer import GraphLayer
import pickle

SAVE_PATH = "state_multilayer.pkl"


model = nn.Sequential(
    GraphLayer(2, 5),
    GraphLayer(5, 5),
    GraphLayer(5, 1),
)

opt = torch.optim.Adam(model.parameters(), lr=0.01)
step = 0

if os.path.exists(SAVE_PATH):
    print(f"loading from {SAVE_PATH}")
    with open(SAVE_PATH, "rb") as f:
        obj = pickle.load(f)
    step = obj["step"]
    model.load_state_dict(obj["model"])
    opt.load_state_dict(obj["opt"])

batch = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).float()
targets = torch.tensor([0, 1, 1, 0]).float()
while True:
    loss = (model(batch).squeeze(-1) - targets).pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"step {step}: loss={loss}")
    step += 1
    if step % 10 == 0:
        with open(SAVE_PATH, "wb") as f:
            pickle.dump(
                dict(
                    step=step,
                    model=model.state_dict(),
                    opt=opt.state_dict(),
                ),
                f,
            )
