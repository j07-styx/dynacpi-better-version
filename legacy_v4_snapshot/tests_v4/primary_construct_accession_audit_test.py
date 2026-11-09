"""Crosswalk completeness and explicit construct verification limitations."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parents[1]/'scripts/audit_primary_construct_accessions_dev.py'
spec=importlib.util.spec_from_file_location('construct_crosswalk',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
r=m.report
assert r['selected_primary_proteins']==16
assert r['mapped_structure_proteins']==13
assert r['missing_structure_proteins']==['IRAK3','RIOK2','SRPK3']
assert len(r['records'])==13
assert all(not x['construct_verified'] for x in r['records'])
assert all(x['primary_accession'] and x['alphafold_uniprot_accession'] for x in r['records'])
print('PASS primary NCBI-to-UniProt accession crosswalk and unverified construct status')
