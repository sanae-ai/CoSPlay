from __future__ import annotations

from collections import defaultdict


def initial_round_key() -> str:
    return "initial_generation"


def self_play_round_key(round_num=None) -> str:
    if round_num is None:
        return "self_play"
    return f"round_{int(round_num)}"


def _round_sort_key(round_key: str):
    if round_key == initial_round_key():
        return (0, 0)
    if round_key.startswith("round_"):
        try:
            return (1, int(round_key.split("_", 1)[1]))
        except Exception:
            return (1, 10**9)
    return (2, round_key)


def _flatten_texts(items):
    if items is None:
        return []
    if isinstance(items, (list, tuple, set)):
        values = []
        for item in items:
            values.extend(_flatten_texts(item))
        return values
    text = str(items).strip()
    return [text] if text else []


class UsageTracker:
    def __init__(self, tokenizer, num_problems: int):
        self.tokenizer = tokenizer
        self.num_problems = max(int(num_problems), 0)
        self.problem_stats = [self._new_problem_stats(i) for i in range(self.num_problems)]
        self.pools = defaultdict(lambda: defaultdict(list))

    @staticmethod
    def _new_problem_stats(problem_idx: int):
        return {
            "problem_index": problem_idx,
            "calls": 0,
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "rounds": {},
        }

    @staticmethod
    def _new_round_stats():
        return {
            "calls": 0,
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "stages": {},
        }

    @staticmethod
    def _new_stage_stats():
        return {
            "calls": 0,
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
        }

    def _count_tokens(self, text) -> int:
        if text is None:
            return 0
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _ensure_problem(self, problem_idx: int):
        while problem_idx >= len(self.problem_stats):
            self.problem_stats.append(self._new_problem_stats(len(self.problem_stats)))

    def _ensure_round(self, problem_idx: int, round_key: str):
        self._ensure_problem(problem_idx)
        rounds = self.problem_stats[problem_idx]["rounds"]
        if round_key not in rounds:
            rounds[round_key] = self._new_round_stats()
        return rounds[round_key]

    def _apply(
        self,
        problem_idx: int,
        round_key: str,
        stage: str,
        calls: int = 0,
        prompt_tokens: float = 0.0,
        completion_tokens: float = 0.0,
    ):
        self._ensure_problem(problem_idx)
        stats = self.problem_stats[problem_idx]
        round_stats = self._ensure_round(problem_idx, round_key)
        stage_stats = round_stats["stages"].setdefault(stage, self._new_stage_stats())
        total_tokens = float(prompt_tokens) + float(completion_tokens)

        stats["calls"] += int(calls)
        stats["prompt_tokens"] += float(prompt_tokens)
        stats["completion_tokens"] += float(completion_tokens)
        stats["total_tokens"] += total_tokens

        round_stats["calls"] += int(calls)
        round_stats["prompt_tokens"] += float(prompt_tokens)
        round_stats["completion_tokens"] += float(completion_tokens)
        round_stats["total_tokens"] += total_tokens

        stage_stats["calls"] += int(calls)
        stage_stats["prompt_tokens"] += float(prompt_tokens)
        stage_stats["completion_tokens"] += float(completion_tokens)
        stage_stats["total_tokens"] += total_tokens

    def record_direct(self, problem_indices, prompts, outputs, stage: str, round_key: str):
        if not prompts:
            return
        for problem_idx, prompt, output in zip(problem_indices, prompts, outputs):
            self._apply(
                int(problem_idx),
                round_key,
                stage,
                calls=1,
                prompt_tokens=self._count_tokens(prompt),
                completion_tokens=self._count_tokens(output),
            )

    def reserve_items(
        self,
        problem_indices,
        prompts,
        outputs,
        stage: str,
        round_key: str,
        items_by_call,
    ):
        if not prompts:
            return
        for problem_idx, prompt, output, items in zip(problem_indices, prompts, outputs, items_by_call):
            problem_idx = int(problem_idx)
            item_texts = _flatten_texts(items)
            prompt_tokens = self._count_tokens(prompt)
            self._apply(
                problem_idx,
                round_key,
                stage,
                calls=1,
                prompt_tokens=float(prompt_tokens),
                completion_tokens=0.0,
            )
            if not item_texts:
                continue
            item_weights = [self._count_tokens(item_text) for item_text in item_texts]
            total_weight = sum(item_weights)
            if total_weight <= 0:
                item_weights = [1] * len(item_texts)
                total_weight = len(item_texts)
            for item_text, weight in zip(item_texts, item_weights):
                self.pools[problem_idx][stage].append(
                    {
                        "text": item_text,
                        "prompt_tokens": 0.0,


                        "completion_tokens": float(self._count_tokens(item_text)),
                        "generated_round": round_key,
                        "consumed_round": None,
                        "used": False,
                    }
                )

    def consume_items(self, problem_idx: int, stage: str, items, round_key: str) -> int:
        item_texts = _flatten_texts(items)
        if not item_texts:
            return 0
        consumed = 0
        pool = self.pools[int(problem_idx)].get(stage, [])
        for item_text in item_texts:
            for entry in pool:
                if entry["used"]:
                    continue
                if entry["text"] != item_text:
                    continue
                entry["used"] = True
                entry["consumed_round"] = round_key
                self._apply(
                    int(problem_idx),
                    round_key,
                    stage,
                    calls=0,
                    prompt_tokens=entry["prompt_tokens"],
                    completion_tokens=entry["completion_tokens"],
                )
                consumed += 1
                break
        return consumed

    def problem_payload(self, problem_idx: int):
        self._ensure_problem(problem_idx)
        stats = self.problem_stats[problem_idx]
        rounds_payload = {}
        for round_key in sorted(stats["rounds"], key=_round_sort_key):
            round_stats = stats["rounds"][round_key]
            rounds_payload[round_key] = {
                "calls": round_stats["calls"],
                "prompt_tokens": round(round_stats["prompt_tokens"], 4),
                "completion_tokens": round(round_stats["completion_tokens"], 4),
                "total_tokens": round(round_stats["total_tokens"], 4),
                "stages": {
                    stage: {
                        "calls": stage_stats["calls"],
                        "prompt_tokens": round(stage_stats["prompt_tokens"], 4),
                        "completion_tokens": round(stage_stats["completion_tokens"], 4),
                        "total_tokens": round(stage_stats["total_tokens"], 4),
                    }
                    for stage, stage_stats in sorted(round_stats["stages"].items())
                    if stage_stats["calls"] or stage_stats["total_tokens"]
                },
            }

        pending_reserved = {}
        for stage, entries in self.pools.get(problem_idx, {}).items():
            remaining = sum(1 for entry in entries if not entry["used"])
            if remaining:
                pending_reserved[stage] = remaining

        return {
            "calls": stats["calls"],
            "prompt_tokens": round(stats["prompt_tokens"], 4),
            "completion_tokens": round(stats["completion_tokens"], 4),
            "total_tokens": round(stats["total_tokens"], 4),
            "rounds": rounds_payload,
            "pending_reserved_items": pending_reserved,
        }

    def sync_to_data(self, data):
        for idx, item in enumerate(data):
            item["model_usage"] = self.problem_payload(idx)
        return data

    def load_from_data(self, data):
        for idx, item in enumerate(data):
            payload = item.get("model_usage")
            if not isinstance(payload, dict):
                continue
            self._ensure_problem(idx)
            stats = self.problem_stats[idx]
            stats["calls"] = int(payload.get("calls", 0) or 0)
            stats["prompt_tokens"] = float(payload.get("prompt_tokens", 0.0) or 0.0)
            stats["completion_tokens"] = float(payload.get("completion_tokens", 0.0) or 0.0)
            stats["total_tokens"] = float(payload.get("total_tokens", 0.0) or 0.0)
            rounds_payload = payload.get("rounds", {})
            if not isinstance(rounds_payload, dict):
                continue
            stats["rounds"] = {}
            for round_key, round_payload in rounds_payload.items():
                if not isinstance(round_payload, dict):
                    continue
                round_stats = self._new_round_stats()
                round_stats["calls"] = int(round_payload.get("calls", 0) or 0)
                round_stats["prompt_tokens"] = float(round_payload.get("prompt_tokens", 0.0) or 0.0)
                round_stats["completion_tokens"] = float(round_payload.get("completion_tokens", 0.0) or 0.0)
                round_stats["total_tokens"] = float(round_payload.get("total_tokens", 0.0) or 0.0)
                stages_payload = round_payload.get("stages", {})
                if isinstance(stages_payload, dict):
                    for stage, stage_payload in stages_payload.items():
                        if not isinstance(stage_payload, dict):
                            continue
                        stage_stats = self._new_stage_stats()
                        stage_stats["calls"] = int(stage_payload.get("calls", 0) or 0)
                        stage_stats["prompt_tokens"] = float(stage_payload.get("prompt_tokens", 0.0) or 0.0)
                        stage_stats["completion_tokens"] = float(stage_payload.get("completion_tokens", 0.0) or 0.0)
                        stage_stats["total_tokens"] = float(stage_payload.get("total_tokens", 0.0) or 0.0)
                        round_stats["stages"][stage] = stage_stats
                stats["rounds"][round_key] = round_stats

    def format_round_summary(self, round_key: str, label: str | None = None):
        lines = []
        header = label or round_key
        lines.append("\n" + "=" * 96)
        lines.append(f"=== Model Usage Summary ({header}) ===")
        lines.append("=" * 96)
        lines.append(
            f"{'Problem':<10} {'RoundCalls':<12} {'RoundTokens':<14} {'CumCalls':<12} {'CumTokens':<14}"
        )
        lines.append("-" * 96)

        total_round_calls = 0
        total_round_tokens = 0.0
        total_calls = 0
        total_tokens = 0.0

        for idx, stats in enumerate(self.problem_stats):
            round_stats = stats["rounds"].get(round_key, self._new_round_stats())
            lines.append(
                f"{idx:<10} {round_stats['calls']:<12} {round_stats['total_tokens']:<14.2f} "
                f"{stats['calls']:<12} {stats['total_tokens']:<14.2f}"
            )
            total_round_calls += round_stats["calls"]
            total_round_tokens += round_stats["total_tokens"]
            total_calls += stats["calls"]
            total_tokens += stats["total_tokens"]

        lines.append("-" * 96)
        lines.append(
            f"{'Average':<10} "
            f"{(total_round_calls / self.num_problems if self.num_problems else 0):<12.2f} "
            f"{(total_round_tokens / self.num_problems if self.num_problems else 0):<14.2f} "
            f"{(total_calls / self.num_problems if self.num_problems else 0):<12.2f} "
            f"{(total_tokens / self.num_problems if self.num_problems else 0):<14.2f}"
        )
        return "\n".join(lines)

    def format_final_summary(self, label: str = "Final"):
        lines = []
        lines.append("\n" + "=" * 96)
        lines.append(f"=== Model Usage Summary ({label}) ===")
        lines.append("=" * 96)
        lines.append(f"{'Problem':<10} {'Calls':<12} {'PromptTok':<14} {'CompTok':<14} {'TotalTok':<14}")
        lines.append("-" * 96)

        total_calls = 0
        total_prompt = 0.0
        total_completion = 0.0
        total_tokens = 0.0

        for idx, stats in enumerate(self.problem_stats):
            lines.append(
                f"{idx:<10} {stats['calls']:<12} {stats['prompt_tokens']:<14.2f} "
                f"{stats['completion_tokens']:<14.2f} {stats['total_tokens']:<14.2f}"
            )
            total_calls += stats["calls"]
            total_prompt += stats["prompt_tokens"]
            total_completion += stats["completion_tokens"]
            total_tokens += stats["total_tokens"]

        lines.append("-" * 96)
        avg_calls = total_calls / self.num_problems if self.num_problems else 0.0
        avg_prompt = total_prompt / self.num_problems if self.num_problems else 0.0
        avg_completion = total_completion / self.num_problems if self.num_problems else 0.0
        avg_total = total_tokens / self.num_problems if self.num_problems else 0.0
        lines.append(
            f"{'Average':<10} {avg_calls:<12.2f} {avg_prompt:<14.2f} "
            f"{avg_completion:<14.2f} {avg_total:<14.2f}"
        )
        return "\n".join(lines)


def init_usage_tracker(args, tokenizer, data):
    tracker = UsageTracker(tokenizer, len(data))
    tracker.load_from_data(data)
    setattr(args, "_usage_tracker", tracker)
    tracker.sync_to_data(data)
    return tracker


def get_usage_tracker(args):
    return getattr(args, "_usage_tracker", None)


def record_direct_usage(args, problem_indices, prompts, outputs, stage: str, round_key: str):
    tracker = get_usage_tracker(args)
    if tracker is None:
        return
    tracker.record_direct(problem_indices, prompts, outputs, stage, round_key)


def reserve_item_usage(args, problem_indices, prompts, outputs, stage: str, round_key: str, items_by_call):
    tracker = get_usage_tracker(args)
    if tracker is None:
        return
    tracker.reserve_items(problem_indices, prompts, outputs, stage, round_key, items_by_call)


def consume_item_usage(args, problem_idx: int, stage: str, items, round_key: str) -> int:
    tracker = get_usage_tracker(args)
    if tracker is None:
        return 0
    return tracker.consume_items(problem_idx, stage, items, round_key)


def sync_usage_to_data(args, data):
    tracker = get_usage_tracker(args)
    if tracker is None:
        return data
    return tracker.sync_to_data(data)


def format_usage_round_summary(args, round_key: str, label: str | None = None) -> str:
    tracker = get_usage_tracker(args)
    if tracker is None:
        return ""
    return tracker.format_round_summary(round_key, label=label)


def format_usage_final_summary(args, label: str = "Final") -> str:
    tracker = get_usage_tracker(args)
    if tracker is None:
        return ""
    return tracker.format_final_summary(label=label)
