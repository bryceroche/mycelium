"""
Page Cache + Replay Buffer (v24.4)

Cache graduated cycle pages to focus training compute on later cycles.
Training gets faster as more cycles graduate.

Key mechanisms:
- Per-step accuracy measurement — track each cycle's contribution
- Graduation threshold — per-step acc > 90% or stable for 2 epochs
- Replay buffer — train later cycles against varying quality earlier thinking
- Probabilistic loading — 70% cache, 20% frontier-1, 10% full runs
"""

import random
from typing import Dict, List, Optional, Tuple
import torch


class PageCache:
    """
    Cache early cycle pages that have graduated (high per-step accuracy).
    Focus training compute on later cycles where accuracy is lowest.
    """

    def __init__(self, device: str = "cuda"):
        self.cache: Dict[str, Dict[int, List[torch.Tensor]]] = {}  # problem_hash → {epoch: [pages]}
        self.graduated_up_to: int = 0  # cycles 0..graduated_up_to-1 are cached
        self.device = device

    def store(self, problem_hash: str, pages: List[torch.Tensor], epoch: int):
        """Cache pages from a full training run."""
        if problem_hash not in self.cache:
            self.cache[problem_hash] = {}
        self.cache[problem_hash][epoch] = [p.detach().cpu() for p in pages]

    def load(
        self,
        problem_hash: str,
        up_to_pass: int,
        epoch: Optional[int] = None
    ) -> Optional[List[torch.Tensor]]:
        """
        Load cached pages up to a specific pass.
        If epoch=None, pick a random cached epoch (replay buffer).
        """
        if problem_hash not in self.cache:
            return None

        available_epochs = list(self.cache[problem_hash].keys())
        if not available_epochs:
            return None

        if epoch is None:
            epoch = random.choice(available_epochs)  # replay buffer
        elif epoch not in available_epochs:
            epoch = max(available_epochs)  # use most recent

        pages = self.cache[problem_hash][epoch][:up_to_pass]
        return [p.to(self.device).requires_grad_(False) for p in pages]

    def update_graduation(
        self,
        per_step_acc: List[float],
        threshold: float = 0.90
    ):
        """Advance the graduation frontier based on per-step accuracy."""
        for i, acc in enumerate(per_step_acc):
            if acc >= threshold:
                self.graduated_up_to = i + 1
            else:
                break
        if self.graduated_up_to > 0:
            print(f"Graduated cycles: 0-{self.graduated_up_to - 1}")

    def clear(self):
        """Clear all cached pages."""
        self.cache.clear()
        self.graduated_up_to = 0

    def stats(self) -> Dict:
        """Return cache statistics."""
        total_problems = len(self.cache)
        total_entries = sum(len(epochs) for epochs in self.cache.values())
        return {
            "problems_cached": total_problems,
            "total_entries": total_entries,
            "graduated_up_to": self.graduated_up_to,
        }


class ReplayBuffer(PageCache):
    """
    Extended PageCache with epoch eviction and diverse loading.
    Stores pages from multiple epochs so later cycles train against
    varying quality earlier thinking — builds robustness.
    """

    def __init__(self, max_epochs_stored: int = 5, device: str = "cuda"):
        super().__init__(device)
        self.max_epochs = max_epochs_stored

    def store(self, problem_hash: str, pages: List[torch.Tensor], epoch: int):
        """Store with eviction of old epochs."""
        super().store(problem_hash, pages, epoch)

        # Evict oldest epochs if too many stored
        if problem_hash in self.cache:
            epochs = sorted(self.cache[problem_hash].keys())
            while len(epochs) > self.max_epochs:
                oldest = epochs.pop(0)
                del self.cache[problem_hash][oldest]

    def load_diverse(
        self,
        problem_hash: str,
        up_to_pass: int,
        current_epoch: int
    ) -> Optional[List[torch.Tensor]]:
        """
        Load from a random epoch, biased toward recent but with diversity.
        60% most recent, 40% random older epoch.
        """
        if problem_hash not in self.cache:
            return None

        available = list(self.cache[problem_hash].keys())
        if not available:
            return None

        # 60% most recent, 40% random older epoch
        if random.random() < 0.6:
            epoch = max(available)
        else:
            epoch = random.choice(available)

        pages = self.cache[problem_hash][epoch][:up_to_pass]
        return [p.to(self.device).requires_grad_(False) for p in pages]


def measure_per_step_accuracy(
    model,
    eval_problems: List[Tuple],
    max_passes: int = 5,
    extract_fn=None,
) -> List[float]:
    """
    At each pass, generate answer from current pages. Track when it becomes correct.

    Args:
        model: The thinking model with think_one_pass and generate_from_pages methods
        eval_problems: List of (problem_text, gold_answer) tuples
        max_passes: Number of thinking passes
        extract_fn: Function to extract numeric answer from generation (default: int)

    Returns:
        List of per-step accuracies [acc_pass0, acc_pass1, ..., acc_passN-1]
    """
    if extract_fn is None:
        extract_fn = lambda x: int(x.strip().split()[-1].replace(",", "").rstrip("."))

    per_step_correct = [0] * max_passes
    model.eval()

    with torch.no_grad():
        for problem, gold in eval_problems:
            state_pages = []
            for pass_num in range(max_passes):
                page = model.think_one_pass(problem, state_pages, pass_num)
                state_pages.append(page)

                # Would the current pages produce the right answer?
                try:
                    answer = model.generate_from_pages(state_pages, problem)
                    pred = extract_fn(answer)
                    if pred == gold:
                        per_step_correct[pass_num] += 1
                except (ValueError, IndexError, AttributeError):
                    pass  # extraction failed

    per_step_acc = [c / len(eval_problems) for c in per_step_correct]
    return per_step_acc


def train_step_with_cache(
    model,
    problem: str,
    gold_answer: int,
    cache: PageCache,
    compute_loss_fn,
    max_passes: int = 5,
    current_epoch: int = 0,
) -> Tuple[float, int]:
    """
    Training step that loads from cache probabilistically.

    Args:
        model: The thinking model
        problem: Problem text
        gold_answer: Gold answer for loss computation
        cache: PageCache or ReplayBuffer instance
        compute_loss_fn: Function (model, pages, gold) → loss
        max_passes: Number of thinking passes
        current_epoch: Current epoch number

    Returns:
        (loss_value, start_pass) tuple
    """
    problem_hash = str(hash(problem))
    graduated = cache.graduated_up_to

    # Decide where to start this step
    r = random.random()

    if graduated > 0 and r < 0.7:
        # 70%: start from the graduation frontier (fast — skip graduated cycles)
        cached_pages = cache.load(problem_hash, graduated)
        if cached_pages is not None:
            state_pages = cached_pages
            start_pass = graduated
        else:
            state_pages = []
            start_pass = 0  # cache miss — full run
    elif graduated > 0 and r < 0.9:
        # 20%: start from one cycle before frontier (keep frontier cycle fresh)
        start_from = max(0, graduated - 1)
        cached_pages = cache.load(problem_hash, start_from)
        if cached_pages is not None:
            state_pages = cached_pages
            start_pass = start_from
        else:
            state_pages = []
            start_pass = 0
    else:
        # 10%: full run from scratch (keep early cycles learning)
        state_pages = []
        start_pass = 0

    # Run remaining cycles
    for pass_num in range(start_pass, max_passes):
        page = model.think_one_pass(problem, state_pages, pass_num)
        state_pages.append(page)

    # Loss and backward (gradient only flows to cycles that ran)
    loss = compute_loss_fn(model, state_pages, gold_answer)
    loss.backward()

    # Update cache on full runs
    if start_pass == 0:
        cache.store(problem_hash, state_pages, epoch=current_epoch)

    return loss.item(), start_pass


def populate_cache(
    model,
    train_data: List[Tuple],
    cache: PageCache,
    graduated_up_to: int,
    epoch: int,
    device: str = "cuda",
):
    """
    Populate cache for all training problems up to graduated cycles.
    Call this when graduation advances to a new level.
    """
    print(f"Populating cache for cycles 0-{graduated_up_to - 1}...")
    model.eval()

    with torch.no_grad():
        for problem, _ in train_data:
            problem_hash = str(hash(problem))
            state_pages = []
            for pass_num in range(graduated_up_to):
                page = model.think_one_pass(problem, state_pages, pass_num)
                state_pages.append(page)
            cache.store(problem_hash, state_pages, epoch)

    print(f"Cache populated: {len(train_data)} problems")


class GraduationTracker:
    """
    Track per-step accuracy across epochs to determine when cycles graduate.
    Graduation requires: per-step acc > threshold OR stable for N epochs.
    """

    def __init__(
        self,
        max_passes: int = 5,
        threshold: float = 0.90,
        stable_epochs: int = 2,
    ):
        self.max_passes = max_passes
        self.threshold = threshold
        self.stable_epochs = stable_epochs
        self.history: List[List[float]] = []  # per-epoch per-step accuracy

    def record(self, per_step_acc: List[float]):
        """Record per-step accuracy for current epoch."""
        self.history.append(per_step_acc)

    def get_graduation_level(self) -> int:
        """
        Determine how many cycles should be graduated.
        Returns the number of cycles (0 to max_passes) that are graduated.
        """
        if len(self.history) < self.stable_epochs:
            return 0

        graduated = 0
        for cycle in range(self.max_passes):
            # Check threshold condition
            if self.history[-1][cycle] >= self.threshold:
                graduated = cycle + 1
                continue

            # Check stability condition
            if len(self.history) >= self.stable_epochs:
                recent = [h[cycle] for h in self.history[-self.stable_epochs:]]
                variance = max(recent) - min(recent)
                if variance < 0.02:  # stable within 2%
                    graduated = cycle + 1
                    continue

            break

        return graduated

    def should_advance(self, cache: PageCache) -> bool:
        """Check if graduation should advance based on history."""
        new_level = self.get_graduation_level()
        return new_level > cache.graduated_up_to


if __name__ == "__main__":
    print("Testing Page Cache + Replay Buffer...")

    # Test PageCache
    cache = PageCache(device="cpu")

    # Simulate storing pages
    pages1 = [torch.randn(64) for _ in range(5)]
    pages2 = [torch.randn(64) for _ in range(5)]

    cache.store("problem_1", pages1, epoch=1)
    cache.store("problem_1", pages2, epoch=2)
    cache.store("problem_2", pages1, epoch=1)

    print(f"Cache stats: {cache.stats()}")

    # Test loading
    loaded = cache.load("problem_1", up_to_pass=3, epoch=1)
    assert loaded is not None
    assert len(loaded) == 3
    print(f"Loaded 3 pages from epoch 1: shapes {[p.shape for p in loaded]}")

    # Test replay (random epoch)
    loaded_random = cache.load("problem_1", up_to_pass=3)
    assert loaded_random is not None
    print(f"Loaded from random epoch: {len(loaded_random)} pages")

    # Test graduation
    per_step_acc = [0.95, 0.92, 0.85, 0.78, 0.72]
    cache.update_graduation(per_step_acc, threshold=0.90)
    assert cache.graduated_up_to == 2
    print(f"Graduated up to: {cache.graduated_up_to} (expected 2)")

    # Test ReplayBuffer
    buffer = ReplayBuffer(max_epochs_stored=3, device="cpu")
    for epoch in range(5):
        buffer.store("problem_x", [torch.randn(64) for _ in range(5)], epoch)

    # Should only have 3 most recent epochs
    assert len(buffer.cache["problem_x"]) == 3
    assert min(buffer.cache["problem_x"].keys()) == 2
    print(f"ReplayBuffer keeps last 3 epochs: {list(buffer.cache['problem_x'].keys())}")

    # Test GraduationTracker
    tracker = GraduationTracker(max_passes=5, threshold=0.90, stable_epochs=2)
    tracker.record([0.85, 0.75, 0.65, 0.55, 0.50])  # epoch 1
    tracker.record([0.92, 0.82, 0.72, 0.62, 0.55])  # epoch 2
    tracker.record([0.95, 0.88, 0.78, 0.68, 0.60])  # epoch 3

    level = tracker.get_graduation_level()
    print(f"GraduationTracker level after 3 epochs: {level}")

    print("\nAll tests passed!")
