"""Descriptor-term vocabularies for the endoscopic and pathology text streams.

The text streams arrive as short descriptor lists rather than free prose, so a term
vocabulary with an explicit absence token is used instead of a sub-word tokenizer.
The vocabulary is fitted on the development split alone and the absence token is
shared by both streams, which is what lets a record with no endoscopic description
at the point of care be represented rather than dropped.

Ref: Methods Sec. 4.5 (text stream and the absence token); Sec. 4.11 (retention of
records with a missing stream).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .schema import Examination

ABSENT_TERM = "<absent>"
UNKNOWN_TERM = "<unknown>"

ENDOSCOPY_KEYS = ("endoscopy_terms",)
PATHOLOGY_KEYS = ("pathology_terms",)


@dataclass(frozen=True)
class TextVocabulary:
    """Term to index maps for the two text streams."""

    endoscopy: dict[str, int]
    pathology: dict[str, int]

    @property
    def endoscopy_size(self) -> int:
        return len(self.endoscopy)

    @property
    def pathology_size(self) -> int:
        return len(self.pathology)

    @classmethod
    def fit(cls, records: tuple[Examination, ...] | list[Examination], min_count: int = 1) -> TextVocabulary:
        """Build vocabularies from the development split, reserving index 0 for absence."""
        endoscopy_counts: Counter[str] = Counter()
        pathology_counts: Counter[str] = Counter()
        for record in records:
            endoscopy_counts.update(record.endoscopy_terms)
            pathology_counts.update(record.pathology_terms)
        endoscopy = {ABSENT_TERM: 0}
        for term, count in sorted(endoscopy_counts.items()):
            if count >= min_count:
                endoscopy[term] = len(endoscopy)
        endoscopy.setdefault(UNKNOWN_TERM, len(endoscopy))
        pathology = {ABSENT_TERM: 0}
        for term, count in sorted(pathology_counts.items()):
            if count >= min_count:
                pathology[term] = len(pathology)
        pathology.setdefault(UNKNOWN_TERM, len(pathology))
        return cls(endoscopy=endoscopy, pathology=pathology)

    def encode_endoscopy(self, terms: tuple[str, ...]) -> tuple[int, ...]:
        return self._encode(self.endoscopy, terms)

    def encode_pathology(self, terms: tuple[str, ...]) -> tuple[int, ...]:
        return self._encode(self.pathology, terms)

    @staticmethod
    def _encode(vocabulary: dict[str, int], terms: tuple[str, ...]) -> tuple[int, ...]:
        if not terms:
            return (vocabulary[ABSENT_TERM],)
        unknown = vocabulary[UNKNOWN_TERM]
        return tuple(vocabulary.get(term, unknown) for term in terms)


def as_text_inputs(record: Examination, vocabulary: TextVocabulary) -> dict[str, tuple[int, ...]]:
    """Indices for both text streams of one record."""
    return {
        "endoscopy": vocabulary.encode_endoscopy(record.endoscopy_terms),
        "pathology": vocabulary.encode_pathology(record.pathology_terms),
    }
