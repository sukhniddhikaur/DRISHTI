from drishti.core.domain import merged_pack
from drishti.core.models import Modality
from drishti.extract.extractor import DocumentExtractor
from drishti.extract.identifiers import find_identifiers, gstin_checksum, valid_cin, valid_gstin, valid_pan, verhoeff_ok
from drishti.extract.ner import NER
from drishti.extract.temporal import find_dates, interval_for
from drishti.understand.parser import parse_text
from drishti.understand.segment import sentences


def extract(text):
    pd = parse_text(text.encode(), "D")
    return DocumentExtractor(merged_pack()).run({"document_id": "D", "case_id": "C", "publication_date": "2025-01-01"}, pd)


def triples(out):
    return {(c.predicate, c.subject_text, c.object_text, c.modality.value) for c in out.claims if c.predicate != "MENTIONED_WITH"}


def test_identifier_validators():
    base = "27AAACR5055K1Z"
    assert valid_gstin(base + gstin_checksum(base))
    assert not valid_gstin(base + ("0" if gstin_checksum(base) != "0" else "1"))
    assert valid_pan("ABCPS1234K") and not valid_pan("ABCXS1234K")  # 4th char must be a holder type
    assert valid_cin("U45200MH2011PTC123456") and not valid_cin("U45200XX2011PTC123456")
    assert verhoeff_ok("2363")  # Verhoeff check digit 3 for "236"


def test_identifiers_and_money():
    t = "Paid Rs. 4.5 crore and ₹12,34,567 to A/c No. 123456789012; call 98765 43210. Vehicle MH 12 AB 1234."
    got = {(m.label, m.normalized) for m in find_identifiers(t)}
    assert ("MONEY", "45000000") in got and ("MONEY", "1234567") in got
    assert ("ACCOUNT", "123456789012") in got and ("PHONE", "+919876543210") in got and ("VEHICLE", "MH12AB1234") in got


def test_utr_is_not_a_phone():
    assert not [m for m in find_identifiers("UTR9876543210 NEFT") if m.label == "PHONE"]


def test_dates_and_intervals():
    ds = find_dates("on 12 March 2025, 03/04/2022, March 5, 2021 and FY 2023-24")
    assert [d.iso for d in ds] == ["2025-03-12", "2022-04-03", "2021-03-05", "2023-04-01"]
    s = "X was appointed as director w.e.f. 01/04/2022 and resigned on 3 May 2023."
    vf, vt, _ = interval_for(s, find_dates(s), 0)
    assert (vf, vt) == ("2022-04-01", "2023-05-03")


def test_sentence_split_keeps_abbreviations():
    t = "Mr. R. K. Sharma joined ABC Pvt. Ltd. on 1 Jan 2020. Smt. Rao left w.e.f. 01/02/2021. ABC Ltd. Mr. X spoke."
    got = [t[a:b] for a, b in sentences(t)]
    assert got[0].endswith("2020.") and got[1].startswith("Smt.") and got[-1] == "Mr. X spoke."


def test_ner_spans_and_attachment():
    t = "Mr. Rajesh Kumar Sharma (DIN: 01234567, PAN ABCPS1234K) is a director of ABC Infra Private Limited."
    sp = {s.label: s for s in NER().extract(t)}
    assert sp["PERSON"].text == "Rajesh Kumar Sharma"  # honorific excluded from span
    assert sp["PERSON"].attrs["identifiers"] == {"DIN": ["01234567"], "PAN": ["ABCPS1234K"]}
    assert sp["ORGANISATION"].normalized == "abc infrastructure pvt ltd"


def test_kinship_reference_is_not_owner():
    t = "Accused Ramesh Yadav @ Pappu, s/o Mohan Yadav, uses mobile number 9876543210."
    per = {s.text: s for s in NER().extract(t) if s.label == "PERSON"}
    assert per["Ramesh Yadav"].attrs["aliases"] == ["Pappu"]
    assert per["Mohan Yadav"].attrs.get("kin_reference")
    assert ("USES_PHONE", "Ramesh Yadav", "9876543210", "asserted") in triples(extract(t))


def test_relation_direction_active_passive_bypassive():
    t = ("ABC Builders Pvt Ltd received a contract from the Public Works Department on 12 March 2025. "
         "The contract was awarded to Orion Constructions Pvt Ltd by the Municipal Corporation of Pune. "
         "Zenith Traders Pvt Ltd appointed Shri Anil Rao as an additional director on 2 May 2021.")
    tr = triples(extract(t))
    assert ("AWARDED_CONTRACT_TO", "Public Works Department", "ABC Builders Pvt Ltd", "asserted") in tr
    assert ("AWARDED_CONTRACT_TO", "Municipal Corporation of Pune", "Orion Constructions Pvt Ltd", "asserted") in tr
    assert ("DIRECTOR_OF", "Anil Rao", "Zenith Traders Pvt Ltd", "asserted") in tr


def test_modality_negation_hedge_report_scope():
    t = ("Mr. Anil Kapoor is not a director of Zenith Builders Pvt Ltd. "
         "It is alleged that Sunrise Traders Pvt Ltd is a subsidiary of Orion Industries Pvt Ltd. "
         "According to the filing, Ms. Priya Nair is a director of Orion Industries Pvt Ltd. "
         "Mr. Arun Joshi, who was never charged, is a director of Lotus Exports Pvt Ltd.")
    mods = {(c.predicate, c.subject_text): c.modality for c in extract(t).claims}
    assert mods[("DIRECTOR_OF", "Anil Kapoor")] == Modality.NEGATED
    assert mods[("SUBSIDIARY_OF", "Sunrise Traders Pvt Ltd")] == Modality.HEDGED
    assert mods[("DIRECTOR_OF", "Priya Nair")] == Modality.REPORTED
    assert mods[("DIRECTOR_OF", "Arun Joshi")] == Modality.ASSERTED  # negation scoped to the relative clause


def test_cooccurrence_is_weak_not_a_relationship():
    out = extract("Mr. Arun Joshi and Ms. Kavita Rao were listed in the cause list.")
    assert {c.predicate for c in out.claims} == {"MENTIONED_WITH"}
    assert all(c.confidence < 0.3 for c in out.claims)


def test_every_claim_span_matches_source():
    t = "Mr. Vikram Mehta resigned as director of Sunrise Traders Pvt Ltd on 3 May 2023. ABC Pvt Ltd acquired XYZ Pvt Ltd."
    pd = parse_text(t.encode(), "D")
    out = DocumentExtractor(merged_pack()).run({"document_id": "D", "case_id": "C"}, pd)
    assert out.claims and all(pd.text[c.char_start:c.char_end] == c.evidence_text for c in out.claims)
    d = next(c for c in out.claims if c.predicate == "DIRECTOR_OF")
    assert d.valid_to == "2023-05-03"


def test_structured_cdr_and_bank(tmp_path):
    from drishti.understand.parser import parse_tabular
    cdr = b"Calling Number,Called Number,Call Date,Duration\n9876543210,9123456780,2024-01-05,60\n"
    pd = parse_tabular(cdr, "D", "text/csv")
    out = DocumentExtractor(merged_pack()).run({"document_id": "D", "case_id": "C"}, pd)
    c = [x for x in out.claims if x.predicate == "CALLED"]
    assert len(c) == 1 and c[0].epistemic.value == "stated" and c[0].valid_from == "2024-01-05"
    bank = b"Txn Date,From Account,To Account,Amount,UTR\n2024-02-01,111122223333,444455556666,49000,UTR1\n"
    out = DocumentExtractor(merged_pack()).run({"document_id": "D", "case_id": "C"}, parse_tabular(bank, "D", "text/csv"))
    t = [x for x in out.claims if x.predicate == "TRANSFERRED_TO"][0]
    assert t.attributes["amount_inr"] == 49000


def test_number_abbreviation_is_not_negation():
    out = extract("The contract under Tender No. PWD/2023/452 was awarded to Aakash Enterprises Pvt Ltd by Public Works Department on 2 March 2023.")
    c = next(c for c in out.claims if c.predicate == "AWARDED_CONTRACT_TO")
    assert c.modality == Modality.ASSERTED and c.subject_text == "Public Works Department"
