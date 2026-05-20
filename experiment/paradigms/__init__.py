"""Paradigm registry."""

from paradigms.base import Paradigm
from paradigms.rsvp_attention import RSVPAttentionParadigm

PARADIGM_REGISTRY: dict[str, type[Paradigm]] = {
    "rsvp_attention": RSVPAttentionParadigm,
}
