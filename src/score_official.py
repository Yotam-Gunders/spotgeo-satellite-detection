import sys
import json
import jsonschema
from pipeline import ROOT, DATA

# validation.py comes from the official starter kit (zenodo.org/records/3874368), not from this repo.
# Its schema is written for draft 7, and recent jsonschema versions default to 2020-12, so pin the validator.
kit = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "official_kit")
sys.path.insert(0, kit)
import validation

jsonschema.validate = lambda inst, schema: jsonschema.Draft7Validator(schema).validate(inst)

sub = ROOT / "submission.json"
validation.validate_json(json.load(open(sub)))
score, mse = validation.compute_score(sub, DATA / "test_anno.json")
print(f"valid submission   1-F1 {score:.4f}   F1 {1 - score:.4f}   MSE {mse:.2f}")
