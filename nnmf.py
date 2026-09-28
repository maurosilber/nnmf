from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class NNMF:
    W: NDArray
    H: NDArray
    X: NDArray
    eps: float = 0
    do_w: bool = True
    do_h: bool = True

    def step_w(self):
        W = self.W
        H = self.H

        if self.do_w:
            hht = H @ H.T
            xht = self.X @ H.T
            W = W * (xht / (W @ hht + self.eps))

        if self.do_h:
            wtw = W.T @ W
            wtx = W.T @ self.X
            H = H * (wtx / (wtw @ H + self.eps))

        # normalize
        scale = H.sum(1, keepdims=True)
        W *= scale[:, 0]
        H /= scale

        self.W = W
        self.H = H

    def loop(
        self,
        W: NDArray,
        H: NDArray,
        *,
        iterations: int,
        intermediate_steps: bool = False,
    ):
        if intermediate_steps:
            Ws = []
            Hs = []

        for _ in range(iterations):
            W, H = self.step(W, H, X)
            if intermediate_steps:
                Ws.append(W)
                Hs.append(H)

        if intermediate_steps:
            return Ws, Hs
        else:
            return W, H
