"""Offline checks for the FIAS/GAR path — no downloads, no lxml required.

Builds a tiny synthetic GAR region (3 XML files), runs the streaming parser, and
verifies the reconstructed canonical records + the coordinate-join key.

    python tests/test_fias.py        # prints OK / raises on failure
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fias import parse_gar
from src.matching import norm_key, norm_house

ADDR_OBJ = """<?xml version="1.0" encoding="utf-8"?>
<ADDRESSOBJECTS>
  <OBJECT OBJECTID="1"   OBJECTGUID="g-reg"  NAME="Курская"  TYPENAME="обл" LEVEL="1" ISACTUAL="1" ISACTIVE="1"/>
  <OBJECT OBJECTID="10"  OBJECTGUID="g-city" NAME="Курск"    TYPENAME="г"   LEVEL="5" ISACTUAL="1" ISACTIVE="1"/>
  <OBJECT OBJECTID="100" OBJECTGUID="g-str"  NAME="Ленина"   TYPENAME="ул"  LEVEL="8" ISACTUAL="1" ISACTIVE="1"/>
  <OBJECT OBJECTID="101" OBJECTGUID="g-old"  NAME="Старая"   TYPENAME="ул"  LEVEL="8" ISACTUAL="0" ISACTIVE="0"/>
</ADDRESSOBJECTS>
"""

HOUSES = """<?xml version="1.0" encoding="utf-8"?>
<HOUSES>
  <HOUSE OBJECTID="1000" OBJECTGUID="g-h1" HOUSENUM="5"  ADDNUM1="1" ISACTUAL="1" ISACTIVE="1"/>
  <HOUSE OBJECTID="1001" OBJECTGUID="g-h2" HOUSENUM="7"              ISACTUAL="0" ISACTIVE="0"/>
  <HOUSE OBJECTID="1002" OBJECTGUID="g-h3" HOUSENUM="12"             ISACTUAL="1" ISACTIVE="1"/>
</HOUSES>
"""

HIER = """<?xml version="1.0" encoding="utf-8"?>
<ITEMS>
  <ITEM OBJECTID="1000" PATH="1.10.100.1000" ISACTIVE="1"/>
  <ITEM OBJECTID="1001" PATH="1.10.100.1001" ISACTIVE="1"/>
  <ITEM OBJECTID="1002" PATH="1.10.1002"     ISACTIVE="1"/>
</ITEMS>
"""


def _write_region(d):
    with open(os.path.join(d, "AS_ADDR_OBJ_1.XML"), "w", encoding="utf-8") as f:
        f.write(ADDR_OBJ)
    with open(os.path.join(d, "AS_HOUSES_1.XML"), "w", encoding="utf-8") as f:
        f.write(HOUSES)
    with open(os.path.join(d, "AS_ADM_HIERARCHY_1.XML"), "w", encoding="utf-8") as f:
        f.write(HIER)


def test_parse_gar():
    with tempfile.TemporaryDirectory() as d:
        _write_region(d)
        recs = parse_gar(d)

    by_guid = {r["fias_guid"]: r for r in recs}
    assert "g-h2" not in by_guid, "inactive house must be skipped"
    assert len(recs) == 2, f"expected 2 houses, got {len(recs)}"

    h1 = by_guid["g-h1"]
    assert h1["region"] == "Курская обл", h1["region"]
    assert h1["city"] == "Курск", h1["city"]          # bare name for the join
    assert h1["street"] == "Ленина", h1["street"]
    assert h1["house"] == "5" and h1["korp"] == "1"
    assert h1["text"] == "Курская обл, г Курск, ул Ленина, д. 5, к. 1", h1["text"]
    assert h1["lat"] is None and h1["lon"] is None    # GAR has no coords

    h3 = by_guid["g-h3"]                               # house hangs under city, no street
    assert h3["street"] is None
    assert h3["text"] == "Курская обл, г Курск, д. 12", h3["text"]
    print("OK test_parse_gar")


POSTAL_ADDR = """<?xml version="1.0" encoding="utf-8"?>
<ADDRESSOBJECTS>
  <OBJECT OBJECTID="1"   OBJECTGUID="g-reg"  NAME="Курская" TYPENAME="обл." LEVEL="1" ISACTUAL="1" ISACTIVE="1"/>
  <OBJECT OBJECTID="10"  OBJECTGUID="g-city" NAME="Курск"   TYPENAME="г."   LEVEL="5" ISACTUAL="1" ISACTIVE="1"/>
  <OBJECT OBJECTID="100" OBJECTGUID="g-str"  NAME="Ленина"  TYPENAME="ул."  LEVEL="8" ISACTUAL="1" ISACTIVE="1"/>
</ADDRESSOBJECTS>
"""
POSTAL_HOUSES = """<?xml version="1.0" encoding="utf-8"?>
<HOUSES>
  <HOUSE OBJECTID="2000" OBJECTGUID="g-h4" HOUSENUM="10" HOUSETYPE="2" ADDNUM1="3" ADDTYPE1="2" ISACTUAL="1" ISACTIVE="1"/>
</HOUSES>
"""
POSTAL_HIER = """<?xml version="1.0" encoding="utf-8"?>
<ITEMS><ITEM OBJECTID="2000" PATH="1.10.100.2000" ISACTIVE="1"/></ITEMS>
"""
POSTAL_PARAMS = """<?xml version="1.0" encoding="utf-8"?>
<PARAMS>
  <PARAM OBJECTID="2000" TYPEID="5" VALUE="305000" CHANGEIDEND="0"/>
  <PARAM OBJECTID="2000" TYPEID="5" VALUE="305999" CHANGEIDEND="123"/>
  <PARAM OBJECTID="2000" TYPEID="8" VALUE="46:29:0:1" CHANGEIDEND="0"/>
</PARAMS>
"""


def test_postal_types_regioncode():
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "46")           # region folder named by its code
        os.makedirs(d)
        for name, body in [("AS_ADDR_OBJ_1.XML", POSTAL_ADDR),
                           ("AS_HOUSES_1.XML", POSTAL_HOUSES),
                           ("AS_ADM_HIERARCHY_1.XML", POSTAL_HIER),
                           ("AS_HOUSES_PARAMS_1.XML", POSTAL_PARAMS)]:
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(body)
        recs = parse_gar(root, want_postal=True)

    assert len(recs) == 1, recs
    r = recs[0]
    assert r["postal"] == "305000", r["postal"]          # current row, not the ended one
    assert r["region_code"] == "46", r["region_code"]
    assert r["korp"] is None                              # ADDTYPE1=2 is строение, not корпус
    assert r["text"] == "305000, Курская обл., г. Курск, ул. Ленина, д. 10, стр. 3", r["text"]
    print("OK test_postal_types_regioncode")


def test_norm_house():
    assert norm_house("5", "1") == "5к1"
    assert norm_house("12к1") == "12к1"
    assert norm_house("12/1") == "12к1"
    assert norm_house("12 к 1") == "12к1"
    assert norm_house("12а") == "12а"                 # litera, not korp
    assert norm_house("5") == "5"
    print("OK test_norm_house")


def test_norm_key_join():
    # FIAS side (bare names, explicit korp) must match OSM side (typed street)
    fias = norm_key("Курск", "Ленина", "5", "1")
    osm = norm_key("г Курск", "улица Ленина", "5к1")
    assert fias == osm == ("курск", "ленина", "5к1"), (fias, osm)
    # hyphenated type abbrev and multi-hyphen city survive
    assert norm_key("", "пр-кт Ленина", "1")[1] == "ленина"
    assert norm_key("Ростов-на-Дону", "Мира", "1")[0] == "ростов-на-дону"
    print("OK test_norm_key_join")


if __name__ == "__main__":
    test_parse_gar()
    test_postal_types_regioncode()
    test_norm_house()
    test_norm_key_join()
    print("\nALL PASSED")
