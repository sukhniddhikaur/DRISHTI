from drishti.evidence.support import aggregate, contradictions, origin_groups
from drishti.core.domain import merged_pack
from drishti.resolve.engine import Resolver
from drishti.resolve.normalize import indic_phonetic, initials_key, org_parts
from drishti.resolve.similarity import jaro_winkler


def m(mid, doc, label, text, norm, **attrs):
    return {"mention_id": mid, "document_id": doc, "label": label, "text": text, "normalized": norm, "attributes": attrs, "char_start": 0}


def test_normalisation_and_phonetics():
    assert org_parts("M/s ABC Infra Pvt. Ltd.") == ("abc infrastructure", "pvt ltd")
    assert org_parts("ABC INFRASTRUCTURE PRIVATE LIMITED") == ("abc infrastructure", "pvt ltd")
    assert indic_phonetic("Mohammad") == indic_phonetic("Muhammad") == indic_phonetic("Mohammed")
    assert initials_key("Shri R. K. Sharma") == initials_key("Rajesh Kumar Sharma") == "rk sharma"
    assert abs(jaro_winkler("martha", "marhta") - 0.9611) < 1e-3


def test_resolver_merges_variants_and_respects_identifier_conflict():
    ms = [m("a", "d1", "ORGANISATION", "ABC Infra Pvt Ltd", "abc infrastructure pvt ltd"),
          m("b", "d2", "ORGANISATION", "ABC Infrastructure Private Limited", "abc infrastructure pvt ltd"),
          m("c", "d3", "PERSON", "Rajesh Kumar Sharma", "rajesh kumar sharma", identifiers={"PAN": ["ABCPS1234K"]}),
          m("d", "d4", "PERSON", "R. K. Sharma", "r k sharma", identifiers={"PAN": ["ABCPS1234K"]}),
          m("e", "d5", "PERSON", "Rajesh Kumar Sharma", "rajesh kumar sharma", identifiers={"PAN": ["ZZZPS9999K"]})]
    ents, mapping, decisions = Resolver().resolve("C", ms, {})
    assert mapping["a"][0] == mapping["b"][0]
    assert mapping["c"][0] == mapping["d"][0]          # shared PAN
    assert mapping["e"][0] != mapping["c"][0]          # same name, conflicting PAN -> cannot-link
    assert any(d["outcome"] == "cannot_link" for d in decisions)


def test_no_naive_transitive_chaining():
    # A~B and B~C are each plausible, but A and C differ in legal form -> must not end up together
    ms = [m("a", "d1", "ORGANISATION", "Zenith Builders Pvt Ltd", "zenith builders pvt ltd"),
          m("b", "d2", "ORGANISATION", "Zenith Builders", "zenith builders"),
          m("c", "d3", "ORGANISATION", "Zenith Builders LLP", "zenith builders llp")]
    _, mapping, _ = Resolver().resolve("C", ms, {})
    assert mapping["a"][0] != mapping["c"][0]


def test_analyst_constraints_are_replayed():
    ms = [m("a", "d1", "PERSON", "Amit Shah", "amit shah"), m("b", "d2", "PERSON", "Amit Shah", "amit shah")]
    _, mp, _ = Resolver().resolve("C", ms, {})
    assert mp["a"][0] == mp["b"][0]
    _, mp, _ = Resolver().resolve("C", ms, {}, analyst=[{"left_id": "a", "right_id": "b", "outcome": "analyst_split"}])
    assert mp["a"][0] != mp["b"][0]


def _claim(cid, doc, text, modality="asserted", conf=0.9, **kw):
    return {"claim_id": cid, "document_id": doc, "evidence_text": text, "modality": modality, "confidence": conf, "status": "unreviewed",
            "epistemic": "extracted", "attributes": kw.pop("attributes", {}), **kw}


def test_syndicated_copies_count_once():
    t = "According to the tender record, the Public Works Department awarded the contract to Orion Builders Pvt Ltd for Rs. 2 crore."
    docs = {"n1": {"document_id": "n1", "reliability": "journalism"}, "n2": {"document_id": "n2", "reliability": "journalism"},
            "o1": {"document_id": "o1", "reliability": "official_record"}}
    cs = [_claim("1", "n1", t), _claim("2", "n2", "Wire: " + t), _claim("3", "o1", "PWD issued a work order to Orion Builders Pvt Ltd.")]
    groups = origin_groups(cs, docs)
    assert len(groups) == 2
    agg = aggregate(cs, docs)
    assert agg["independent_sources"] == 2 and agg["supporting_documents"] == 3
    assert agg["belief"] < 1 - (1 - 0.9) ** 3   # not treated as three independent confirmations


def test_negation_lowers_belief_and_contradictions():
    docs = {"a": {"document_id": "a", "reliability": "official_record"}, "b": {"document_id": "b", "reliability": "official_record"}}
    base = dict(subject_entity_id="P", object_entity_id="O", predicate="DIRECTOR_OF", valid_from=None, reported_at=None)
    cs = [_claim("1", "a", "P is a director of O.", valid_to="2023-01-01", **{**base, "reported_at": "2023-02-01"}),
          _claim("2", "b", "P is not a director of O.", modality="negated", valid_to=None, **base),
          _claim("3", "b", "P is a director of O, sources said.", modality="reported", valid_to=None, **{**base, "reported_at": "2024-05-01"})]
    agg = aggregate(cs, docs)
    assert agg["contra"] > 0 and agg["belief"] < agg["support"]
    kinds = {c["type"] for c in contradictions(cs, merged_pack())}
    assert {"polarity", "temporal"} <= kinds
