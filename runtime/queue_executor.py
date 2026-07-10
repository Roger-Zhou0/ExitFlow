"""Queue-based segmented executor with competitive pull scheduling.

This keeps the scheduling simple:
- One inbound queue per segment.
- Each segment has N workers (possibly on the same GPU) that compete for work.
- Forward decisions push the hidden state into the next segment's queue.
- Exit decisions resolve the request and release resources.

Use this when Stage-1 replicas > Stage-2 replicas > Stage-3 replicas. Place
the queue right in front of the narrower stage so upstream workers never need
to pick a specific downstream replica.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Dict, Iterable, Optional

import torch

from .forwarder import ForwarderDecision
from .segment_worker import SegmentWorker


@dataclass
class WorkItem:
    request_id: str
    hidden_state: torch.Tensor
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkResult:
    request_id: str
    segment_id: str
    decision: Optional[ForwarderDecision]
    hidden_state: Optional[torch.Tensor]   # exit 无 logits 时用作最终 hidden；有 logits 时为 None
    logits: Optional[torch.Tensor]         # 早退出时的 1D logits（通常在 CPU）
    step_index: int
    metadata: dict[str, Any]


class RequestHandle:
    """Simple future-like handle to wait for completion."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._event = threading.Event()
        self.result: Optional[WorkResult] = None

    def set_result(self, result: WorkResult) -> None:
        self.result = result
        self._event.set()

    def wait(self, timeout: Optional[float] = None) -> Optional[WorkResult]:
        self._event.wait(timeout)
        return self.result


class SegmentedQueueExecutor:
    """Competitive-pull scheduler across segment replicas."""

    def __init__(
        self,
        segment_workers: Dict[str, Iterable[SegmentWorker]],
        entry_segment: str,
        queue_capacity: int = 32,
        max_hops: Optional[int] = None,
    ) -> None:
        self.entry_segment = entry_segment
        self.max_hops = max_hops
        self.queues: Dict[str, Queue[WorkItem]] = {
            segment_id: Queue(maxsize=queue_capacity) for segment_id in segment_workers.keys()
        }
        self.pools: Dict[str, deque[SegmentWorker]] = {
            segment_id: deque(workers) for segment_id, workers in segment_workers.items()
        }
        self._results: Dict[str, RequestHandle] = {}
        self._hop_counters: Dict[str, int] = defaultdict(int)
        self._shutdown = threading.Event()
        self._threads: list[threading.Thread] = []

        for segment_id in segment_workers.keys():
            thread = threading.Thread(target=self._worker_loop, args=(segment_id,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def submit(self, item: WorkItem) -> RequestHandle:
        handle = RequestHandle(item.request_id)
        self._results[item.request_id] = handle
        self._enqueue(self.entry_segment, item)
        return handle

    def shutdown(self) -> None:
        self._shutdown.set()
        for q in self.queues.values():
            q.put(None)  # type: ignore[arg-type]
        for t in self._threads:
            t.join(timeout=1.0)

    # Internal helpers -----------------------------------------------------
    def _enqueue(self, segment_id: str, item: WorkItem) -> None:
        self.queues[segment_id].put(item)

    def _mark_done(self, result: WorkResult) -> None:
        handle = self._results.get(result.request_id)
        if handle is not None:
            handle.set_result(result)
        # request 结束，清理内部状态
        self._results.pop(result.request_id, None)
        self._hop_counters.pop(result.request_id, None)

    def _worker_loop(self, segment_id: str) -> None:
        queue = self.queues[segment_id]
        pool = self.pools[segment_id]
        while not self._shutdown.is_set():
            item = queue.get()
            if item is None:  # shutdown sentinel
                break

            worker = pool[0]
            pool.rotate(-1)

            # 1) 被 pull 出来的这一份 hidden_state 的所有权
            old_state = item.hidden_state

            # 通过本 segment 计算新 hidden，并交给 forwarder 决策
            hidden_out, decision = worker.run(old_state)

            # old_state 已经用完，可以丢引用
            del old_state

            # hop 计数
            hops = self._hop_counters[item.request_id]
            self._hop_counters[item.request_id] = hops + 1

            # 是否应该在这一 hop 终结这个 request
            exit_or_done = (
                decision is None
                or decision.exit_now
                or (decision is not None and decision.next_segment_id is None)
                or (self.max_hops is not None and hops + 1 >= self.max_hops)
            )

            if exit_or_done:
                # 分两类：
                # 1) decision 有 logits：早退出，直接用 logits，hidden 可以丢掉；
                # 2) 没有 logits（没有 forwarder / gating 没 exit / final segment）：
                #    把 hidden_out 当成最终 hidden，交给上层 lm_head 处理。
                if decision is not None and decision.logits is not None:
                    # 早退出：只返回 logits
                    logits = decision.logits
                    final_hidden: Optional[torch.Tensor] = None
                else:
                    # 没有 logits：需要最终 hidden
                    final_hidden = hidden_out
                    # 为了省 GPU 显存，可以搬到 CPU
                    if isinstance(final_hidden, torch.Tensor) and final_hidden.is_cuda:
                        final_hidden = final_hidden.to("cpu")
                    logits = None

                result = WorkResult(
                    request_id=item.request_id,
                    segment_id=worker.context.segment_id,
                    decision=decision,
                    hidden_state=final_hidden,
                    logits=logits,
                    step_index=item.step_index,
                    metadata=item.metadata,
                )
                self._mark_done(result)

                # 当前 hop 用到的对象全部丢引用，便于 GC/allocator 复用显存
                del hidden_out, decision, item, result
                continue

            # 不终结：继续往下游 segment 转发
            next_segment = decision.next_segment_id if decision is not None else None
            forwarded_state = decision.forwarded_state if decision is not None else None

            if next_segment is None or forwarded_state is None:
                # 理论上不该发生，防御处理
                result = WorkResult(
                    request_id=item.request_id,
                    segment_id=worker.context.segment_id,
                    decision=decision,
                    hidden_state=None,
                    logits=None,
                    step_index=item.step_index,
                    metadata=item.metadata,
                )
                self._mark_done(result)
                del hidden_out, decision, item, result
                continue

            # 路径跟踪（只存 segment_id 字符串，不存 tensor）
            path = list(item.metadata.get("path", []))
            path.append(worker.context.segment_id)

            forwarded_item = WorkItem(
                request_id=item.request_id,
                hidden_state=forwarded_state,  # 新一 hop 的唯一 hidden
                step_index=item.step_index,
                metadata={**item.metadata, "path": path},
            )

            # 旧的 hidden_out / decision / item 引用全部丢掉
            del hidden_out, decision, item

            # 把所有权交给下一个 segment 的队列
            self._enqueue(next_segment, forwarded_item)

    def describe_topology(self) -> Dict[str, Any]:
        """Return a summary of pools for logging/debugging."""
        summary = {}
        for seg_id, pool in self.pools.items():
            devices = [w.context.device for w in pool]
            summary[seg_id] = {"replicas": len(pool), "devices": devices}
        return summary


class AutoRegressiveTracker:
    """Tracks per-request token generation to know when a request is finished."""

    @dataclass
    class _State:
        max_new_tokens: int
        eos_token_id: Optional[int]
        generated: list[int] = field(default_factory=list)
        done: bool = False
        exit_segments: list[str] = field(default_factory=list)

    def __init__(self) -> None:
        self._states: Dict[str, AutoRegressiveTracker._State] = {}
        self._lock = threading.Lock()

    def register(self, request_id: str, *, max_new_tokens: int, eos_token_id: Optional[int]) -> None:
        with self._lock:
            self._states[request_id] = AutoRegressiveTracker._State(
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )

    def record_step(self, request_id: str, token_id: int, exit_segment: str) -> bool:
        """Returns True if the request is now complete."""
        with self._lock:
            state = self._states.get(request_id)
            if state is None or state.done:
                return True
            state.generated.append(token_id)
            state.exit_segments.append(exit_segment)
            if (state.eos_token_id is not None and token_id == state.eos_token_id) or (
                len(state.generated) >= state.max_new_tokens
            ):
                state.done = True
            return state.done

    def result(self, request_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            state = self._states.get(request_id)
            if state is None or not state.done:
                return None
            return {
                "generated_ids": list(state.generated),
                "exit_segments": list(state.exit_segments),
            }
