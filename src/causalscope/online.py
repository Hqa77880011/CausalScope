class StableDecision:
    def __init__(self, windows=3):
        self.windows = windows
        self.top = None
        self.count = 0

    def update(self, row):
        component = row["order"][0]
        if row["abstain"]:
            self.top, self.count = None, 0
            return None
        self.count = self.count + 1 if component == self.top else 1
        self.top = component
        if self.count < self.windows:
            return None
        return {
            "timestamp": row["end"],
            "component": component,
            "metrics": row["strongest_metrics"][component],
            "order": row["order"][:2],
        }


def replay_decisions(rows, windows=3):
    outputs = []
    for run_id in sorted({row["run_id"] for row in rows}):
        decision = StableDecision(windows)
        for row in sorted(
            [row for row in rows if row["run_id"] == run_id], key=lambda row: row["end"]
        ):
            recommendation = decision.update(row)
            if recommendation is not None:
                index = row["components"].index(recommendation["component"])
                recommendation["run_id"] = run_id
                recommendation["root_probability"] = (
                    row["probabilities"][index] if "probabilities" in row else None
                )
                recommendation["replacement_gain"] = (
                    row["gains"][index] if "gains" in row else None
                )
                outputs.append(recommendation)
    return outputs
