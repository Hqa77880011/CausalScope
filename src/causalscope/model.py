import torch
from torch import nn
from torch.nn import functional as F


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight):
        ctx.weight = weight
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.weight * gradient, None


class CausalBlock(nn.Module):
    def __init__(self, inputs, outputs, dilation, dropout):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(inputs, outputs, 3, dilation=dilation)
        self.skip = (
            nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.gelu(self.conv(F.pad(x, (self.padding, 0))))) + self.skip(
            x
        )


class TemporalEncoder(nn.Module):
    def __init__(self, hidden, dropout, temporal):
        super().__init__()
        self.temporal = temporal
        if temporal == "lstm":
            self.lstm = nn.LSTM(3, hidden, batch_first=True)
        else:
            branches = []
            for dilations in ((1, 2, 4), (1, 4, 16)):
                layers = []
                inputs = 3
                for dilation in dilations:
                    layers.append(CausalBlock(inputs, hidden // 2, dilation, dropout))
                    inputs = hidden // 2
                branches.append(nn.Sequential(*layers))
            self.branches = nn.ModuleList(branches)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        batch, nodes, length, features = x.shape
        sequence = x.reshape(batch * nodes, length, features)
        if self.temporal == "lstm":
            h, _ = self.lstm(sequence)
        else:
            h = torch.cat(
                [branch(sequence.transpose(1, 2)) for branch in self.branches], dim=1
            ).transpose(1, 2)
        return self.norm(h).reshape(batch, nodes, length, -1)


class TypedGraphLayer(nn.Module):
    def __init__(self, hidden, edges, dropout):
        super().__init__()
        self.self_weight = nn.Linear(hidden, hidden)
        self.prior_weight = nn.Linear(hidden, hidden, bias=False)
        self.lagged_weight = nn.Linear(hidden, hidden, bias=False)
        self.groups = []
        for relation in ("prior", "lagged"):
            lags = sorted({edge["lag"] for edge in edges if edge["type"] == relation})
            for lag in lags:
                group = [
                    edge
                    for edge in edges
                    if edge["type"] == relation and edge["lag"] == lag
                ]
                name = f"{relation}_{lag}"
                self.register_buffer(
                    f"source_{name}",
                    torch.tensor([edge["source"] for edge in group]),
                    persistent=False,
                )
                self.register_buffer(
                    f"destination_{name}",
                    torch.tensor([edge["destination"] for edge in group]),
                    persistent=False,
                )
                self.register_buffer(
                    f"alpha_{name}",
                    torch.tensor([edge["alpha"] for edge in group]),
                    persistent=False,
                )
                self.groups.append((relation, lag, name))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        output = self.self_weight(h)
        prior = self.prior_weight(h)
        lagged = self.lagged_weight(h)
        for relation, lag, name in self.groups:
            source = getattr(self, f"source_{name}")
            destination = getattr(self, f"destination_{name}")
            alpha = getattr(self, f"alpha_{name}")
            message = (prior if relation == "prior" else lagged)[:, source]
            if lag:
                message = F.pad(message[:, :, :-lag], (0, 0, lag, 0))
            output = output.index_add(
                1, destination, alpha[None, :, None, None] * message
            )
        return self.dropout(F.gelu(output))


class AttentionPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attention = nn.Linear(hidden, 1)

    def forward(self, h):
        weights = self.attention(h).softmax(dim=1)
        return (weights * h).sum(dim=1), weights.squeeze(-1)


class LocalizationModel(nn.Module):
    def __init__(self, schema, graph, domains, config, method="causalscope"):
        super().__init__()
        self.schema = schema
        self.method = method
        hidden = config["hidden"]
        if hidden % 2:
            raise ValueError("Hidden width must be even")
        self.invariant = config["invariant"] and method == "causalscope"
        temporal = "lstm" if method == "lstm" else config["temporal"]
        self.temporal = TemporalEncoder(hidden, config["dropout"], temporal)
        self.use_graph = config["graph"] and method not in ("tcn", "lstm")
        self.node_kind = nn.Embedding(3, hidden)
        kinds = ["component", "phase", "rank"]
        self.register_buffer(
            "kind_ids",
            torch.tensor([kinds.index(metric["kind"]) for metric in schema["metrics"]]),
        )
        self.layers = nn.ModuleList(
            [
                TypedGraphLayer(hidden, graph["edges"], config["dropout"])
                for _ in range(2)
            ]
            if self.use_graph
            else []
        )
        self.component_pool = AttentionPool(hidden)
        self.graph_pool = AttentionPool(hidden)
        self.owned = [
            [
                i
                for i, metric in enumerate(schema["metrics"])
                if metric["owner"] == component
            ]
            for component in schema["components"]
        ]
        self.phase_nodes = [
            i for i, metric in enumerate(schema["metrics"]) if metric["kind"] == "phase"
        ]
        self.root_head = nn.Linear(hidden, 1)
        self.step_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.domain_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, max(domains, 1))
        )
        self.fault_head = nn.Linear(hidden, 1)

    def forward(self, x, reversal_weight=1.0):
        h = self.temporal(x) + self.node_kind(self.kind_ids)[None, :, None, :]
        for layer in self.layers:
            h = layer(h)
        nodes = h[:, :, -1]
        components = torch.stack(
            [self.component_pool(nodes[:, owned])[0] for owned in self.owned], dim=1
        )
        pool_input = torch.cat([components, nodes[:, self.phase_nodes]], dim=1)
        graph, _ = self.graph_pool(pool_input)
        domain_input = (
            GradientReverse.apply(graph, reversal_weight) if self.invariant else graph
        )
        return {
            "root_logits": self.root_head(components).squeeze(-1),
            "step": self.step_head(graph).squeeze(-1),
            "domain_logits": self.domain_head(domain_input),
            "fault_logit": self.fault_head(graph).squeeze(-1),
        }
