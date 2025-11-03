# Raft Learning Lab

This workspace is a hands-on companion for studying the Raft consensus algorithm described in Ongaro & Ousterhout (2014). It blends readings, visualizations, and incremental coding exercises so you can build a working Raft implementation from scratch.

---

## 1. Which language should you use?

Raft can be implemented in any general-purpose language. For learning and rapid iteration:

1. **Python (recommended here)** – Great for prototyping and experimenting with the algorithmic ideas while keeping the code concise. The `src/` tree below is scaffolded for a pure-Python implementation with unit tests driven by `pytest`.
2. **Go** – The reference implementation from the Raft paper is in Go, and the ecosystem has mature libraries if you later want production-grade performance.
3. **Rust or C++** – Ideal once you care deeply about memory layout and performance, but they introduce more upfront friction.

> ✅ **Pick Python first**, get the algorithm right, then port to Go/Rust once you can trace the message/state transitions confidently.

---

## 2. Workspace layout

```
learning/raft/
├── README.md            # You are here
├── resources/           # Key papers, talks, links (curated list)
├── src/                 # Implementation code (Python package scaffolded)
└── tests/               # Pytest-based regression & scenario tests
```

Create a virtual environment for the lab:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip pytest
```

---

## 3. Essential references

| Type | Resource |
|------|----------|
| Paper | Ongaro & Ousterhout, *In Search of an Understandable Consensus Algorithm (Extended Version)* ([raft.github.io](https://raft.github.io/)) |
| Visualization | [The Secret Lives of Data – Raft](http://thesecretlivesofdata.com/raft/) |
| Blog Series | Eli Bendersky, *Implementing Raft* ([Part 0 intro](https://eli.thegreenplace.net/2020/implementing-raft-part-0-introduction/)) |
| Spec | Official TLA+ spec (linked from the paper) – useful for validating edge cases |

Keep these close; revisit them while implementing each module below.

---

## 4. Incremental build plan

Follow the modules in order. Each module maps to code you will add under `src/` plus tests in `tests/`.

### Module 0 – Warm-up
- Skim the Raft paper Sections 2–5.
- Run the visualizations linked above. Pay attention to term numbers, leader transitions, and the log-matching property.
- Jot down the invariants described in Figure 3 of the paper.

### Module 1 – Core data model & event loop
- Define enums for server roles and RPC types.
- Create a `RaftNode` class with in-memory state (current term, votedFor, log, commit index, etc.).
- Sketch the event loop for processing inbound messages and timers. Start with a deterministic “single-threaded simulator” rather than real networking.

### Module 2 – Leader election
- Implement follower → candidate → leader transitions.
- Model randomized election timeouts.
- Write deterministic tests that simulate dropped heartbeats, split votes, and election recovery.

### Module 3 – Log replication
- Encode `AppendEntries` RPCs and the consistency checks.
- Maintain `nextIndex` / `matchIndex` per follower.
- Add tests for conflicting entries and log convergence scenarios (mirror Figure 7 from the paper).

### Module 4 – Commitment & safety
- Enforce the “current-term-only” commit rule.
- Validate the Leader Completeness and State Machine Safety properties with scenario tests.
- Add client command submission and result propagation once entries commit.

### Module 5 – Persistence & crash recovery
- Serialize Raft state to disk (term, vote, log) to survive restarts.
- Simulate crashes and restarts in tests to prove idempotency.

### Module 6 – Snapshotting (optional stretch)
- Implement the snapshot/install-snapshot flow to compact logs.
- Exercise membership change logic once the base algorithm is stable.

---

## 5. Testing philosophy

- **Property-based unit tests**: Validate invariants after each event (message receipt, timer fire).
- **Scenario tests**: Script timelines (e.g., “leader crash during replication”, “recovery with stale followers”).
- **Fuzz harness (bonus)**: Randomize message delays/drops to reveal hidden bugs.

All tests should be deterministic; a simple in-memory “network” queue where you control delivery order makes debugging much easier than using sockets early on.

---

## 6. Next steps

1. Initialize the Python package skeleton in `src/` (see the TODO markers).
2. Write the first Pytest covering a bare `RaftNode` initialization and leader election timeout.
3. Keep notes in `resources/` as you discover tricky invariants or good external explanations.

Happy hacking! Raft rewards careful, incremental progress—treat every new transition as something you can explain back to Figure 2 of the paper.
