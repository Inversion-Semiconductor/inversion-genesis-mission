"""Fixed batch-one network capture; fitted table preprocessing stays eager."""

import torch


class CapturedNetwork:
    def __init__(self, network):
        self.eager = network.forward
        self.entries = {}
        self.replays = 0
        self.enabled = True
        network.forward = self.forward

    def forward(self, x, y, categorical_mask, *, cache=None):
        if not self.enabled or x.shape[-2] != 1:
            return self.eager(x, y, categorical_mask, cache=cache)
        assert cache is not None and cache.is_replaying and y.shape[-1] == 0
        # The table API rebuilds Cache wrappers but reuses these resident tensors.
        # Include every tensor's address so different ensemble caches cannot alias.
        key = (
            tuple(x.shape),
            x.dtype,
            tuple(categorical_mask.shape),
            tuple(t.data_ptr() for t in cache._tensors()),
        )
        if key not in self.entries:
            sx = x.clone()
            sy = y.clone()
            sm = categorical_mask.clone()
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.eager(sx, sy, sm, cache=cache)
            torch.cuda.current_stream(x.device).wait_stream(stream)
            torch.cuda.synchronize(x.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.eager(sx, sy, sm, cache=cache)
            # Retain cache and buffers for the graph's complete lifetime.
            self.entries[key] = (graph, sx, sy, sm, output, cache)
        graph, sx, sy, sm, output, _ = self.entries[key]
        sx.copy_(x)
        sm.copy_(categorical_mask)
        graph.replay()
        self.replays += 1
        # Table postprocessing may mutate outputs; give each call its own tensor.
        return output.clone()

    def summary(self):
        return dict(captured_graphs=len(self.entries), replays=self.replays)
