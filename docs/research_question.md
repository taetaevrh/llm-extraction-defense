# Provisional research question

"Can a novel defense algorithm reduce black-box LLM model extraction effectiveness while preserving utility for legitimate users?"

The word "novel" states a research aspiration. No algorithm has been designed,
and no novelty, effectiveness, or generality claim is made.

## Scope to review

Initially study an authorized, locally hosted teacher through a text-only
black-box interface. The attacker may send bounded queries and observe served
text/denials, but receives no teacher weights, gradients, hidden states, or
logits. The local host may access weights for inference; the attacker interface
must not expose them. Confirm the threat model with the project owner.

Use a small label-based language task to make the initial agreement metric
unambiguous. This is a feasibility study of behavioral imitation on one task,
not evidence of recovering weights or broad LLM capabilities. Establish how
much imitation already exists in the untrained student.

The primary outcome is held-out teacher-student agreement under a fixed query
attempt budget. Compare no defense, random denial at a matched rate, and a
future proposed defense. Measure legitimate-user utility alongside extraction
effectiveness so denying service cannot masquerade as a useful defense.

## Decisions reserved for research review

- Teacher/student IDs, immutable revisions, licenses, and initial capability gap.
- Dataset, language, task labels, split sizes, overlap checks, and prompt format.
- Student adaptation method, compute allocation, and query schedule.
- Defense access/knowledge, state, and attacker adaptation assumptions.
- Denial-rate matching policy, acceptable legitimate utility loss, and the
  minimum meaningful effect; predeclare before comparative testing.
- Related work and novelty assessment before any future novelty claim.
