"""Read-only primary DAVIS accession versus predicted-structure provenance audit."""
import collections
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=json.loads((ROOT/'data/audit/davis_primary_20261107/extracted_tables.json').read_text())
manifest=json.loads((ROOT/'data/engineering_pilot/limited_alpha_3d_graph_manifest.json').read_text())
primary={r[2]:r for r in source['selected_table4_rows']}
records=manifest['records']
assert len(records)==13 and len(primary)==16
items=[]
for r in records:
    name=r['source_name']
    row=primary[name]
    items.append({'protein':name,'primary_accession':row[0],
                  'primary_gene':row[1],
                  'alphafold_uniprot_accession':r['uniprot_accession'],
                  'source_sequence_length':r['source_sequence_length'],
                  'alphafold_reference_length':r['reference_length'],
                  'reference_start_1_based':r['reference_start_1_based'],
                  'reference_end_1_based_inclusive':r['reference_end_1_based_inclusive'],
                  'backbone_coverage':r['backbone_coverage'],
                  'accession_namespace_match':row[0].split('_')[0]==r['uniprot_accession'].split('_')[0],
                  'construct_verified':False})
assert len({r['protein'] for r in items})==13
missing=sorted(set(primary)-{r['protein'] for r in items})
report={'status':'PRIMARY_ACCESSION_CROSSWALK_NOT_CONSTRUCT_VERIFICATION',
        'selected_primary_proteins':len(primary),'mapped_structure_proteins':len(items),
        'missing_structure_proteins':missing,
        'primary_accession_namespace':'NCBI protein accession',
        'structure_accession_namespace':'UniProt',
        'cross_namespace_exact_string_comparison_is_not_identity_verification':True,
        'all_assayed_constructs_unverified':True,
        'records':items,
        'required_next_evidence':['Map NCBI protein accession to UniProt isoform and sequence version',
          'Verify assay reagent/domain residue bounds and mutations against primary methods or vendor constructs',
          'Verify mapped AlphaFold residues correspond to the assayed construct',
          'Freeze a reviewed provenance table before independent evaluation']}
print(json.dumps(report,indent=2))
