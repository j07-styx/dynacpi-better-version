"""Synthetic IRAK3 reviewer-packet tests; no private records."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_irak3_registry_review_packet_dev import make

def main():
    headers=["a","b","c"]+["Compound"+str(i) for i in range(7)]
    table={"table4_headers":headers,"selected_table4_rows":[["","","IRAK3"]+[str(i+1) for i in range(7)]]}
    leads=[{"target":"IRAK3","registry_status":"REGISTRY_LINK_WITH_PRIMARY_COLUMN",
            "review_decision":"UNREVIEWED","structure_approval":False,
            "reactant_set_id":str(i),"primary_compound":"Compound"+str(i),
            "pubchem_cid":str(100+i),"numeric_comparison":"EXACT_NUMERIC_MATCH"} for i in range(7)]
    provenance={str(i):{"BindingDB MonomerID":str(i),"Kd (nM)":str(i+1)} for i in range(7)}
    packet=make(leads,provenance,table)
    assert len(packet)==7 and all(r["review_decision"]=="UNREVIEWED" and r["stereo_verified"]=="NO" for r in packet)
    provenance["0"]["Kd (nM)"]="100"
    try:make(leads,provenance,table)
    except ValueError:pass
    else:raise AssertionError("Numerical mismatch accepted")
    provenance["0"]["Kd (nM)"]="1"
    leads[0]["review_decision"]="APPROVED"
    try:make(leads,provenance,table)
    except ValueError:pass
    else:raise AssertionError("Unexpected approval accepted")
    print("PASS: private packet fields, exact numerical check, approval refusal")
if __name__=="__main__":main()

