"""Read publisher XLS files using an environment with xlrd; no workbook edits."""
import argparse,hashlib,json
from pathlib import Path
import xlrd
def extract(source):
    paths={i:source/("41587_2011_BFnbt1990_MOESM"+str(i)+"_ESM.xls") for i in range(3,7)}
    sheets={i:xlrd.open_workbook(p).sheet_by_index(0) for i,p in paths.items()}
    names=set("CDK8 CLK1 CSNK1G3 MEK6 MKNK2 NEK2 ROCK2 TGFBR1 VRK2 ADCK4 FRK MEK5 OSR1 RIOK2 SRPK3 IRAK3".split())
    s,g,c=sheets[5],sheets[3],sheets[4]
    assert (s.nrows,s.ncols)==(443,75)
    return {"table4_headers":s.row_values(0),
       "selected_table4_rows":[s.row_values(i) for i in range(1,s.nrows) if s.cell_value(i,2) in names],
       "table1_rows":[g.row_values(i) for i in range(g.nrows)],
       "table3_rows":[c.row_values(i) for i in range(c.nrows)],
       "hashes":{str(i):hashlib.sha256(p.read_bytes()).hexdigest() for i,p in paths.items()}}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    with a.output.open("x") as f:json.dump(extract(a.source),f)
