import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import DataFrameLoader
from src.models.coral import Conv1DCoralNet
from src.tasks.ordinal_task import coral_loss, logits_to_ordinal_predictions

RUTA = "dataset/15 atributos R0-R5.sav"
loader = DataFrameLoader()
df = loader.load(RUTA)
X = torch.as_tensor(loader.feature_matrix(df))

for objetivo in ["GDS_R4", "GDS_R5"]:
    y = np.unique(df[objetivo].to_numpy(), return_inverse=True)[1]
    K = int(y.max()) + 1
    y_t = torch.as_tensor(y, dtype=torch.long)
    p = np.array([(y > k).mean() for k in range(K - 1)])
    ideal = np.log(p / (1 - p))
    print(f"\n{objetivo}: sesgos ideales {ideal.round(2)} | clases reales {np.bincount(y)}")

    for con_inicio in (False, True):
        torch.manual_seed(42)
        model = Conv1DCoralNet(15, K, (8, 16), 3, 32, 0.15)
        if con_inicio:
            gaps = -np.diff(ideal)
            with torch.no_grad():
                model.head.first_bias.fill_(float(ideal[0]))
                model.head.bias_gaps.copy_(
                    torch.as_tensor(np.log(np.expm1(gaps)), dtype=torch.float32)
                )

        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        datos = DataLoader(TensorDataset(X, y_t), batch_size=32, shuffle=True)
        model.train()
        for _ in range(20):
            for xb, yb in datos:
                opt.zero_grad()
                coral_loss(model(xb), yb, K).backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            pred = logits_to_ordinal_predictions(model(X)).numpy()
            head = model.head
            sesgos = torch.cat([
                head.first_bias,
                head.first_bias - torch.cumsum(torch.nn.functional.softplus(head.bias_gaps), 0),
            ]).numpy()
        print(f"  inicio prevalencias={con_inicio}: sesgos {sesgos.round(2)} | "
              f"accuracy train {(pred == y).mean():.4f} | predichas {np.bincount(pred, minlength=K)}")