"""Real-image smoke check of the final service, including persisted retrieval."""
from pathlib import Path
import base64, csv, json, sys, tempfile, time
from xml.etree import ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fastapi.testclient import TestClient
from postcode_ml.service import create_app
from postcode_ml.best_model import Predictor

root = Path(__file__).resolve().parents[1]
workspace = root.parent
torch.set_num_threads(4)
photo = root / 'tests/fixtures/cargo_sample.jpg'
bundle = root / 'runtime/attention518'
start = time.perf_counter()
predictor = Predictor(bundle, device='cpu')
with tempfile.TemporaryDirectory() as temp:
    db = Path(temp) / 'checks.sqlite3'
    app = create_app(db_path=db, predictor=predictor)
    body = '<assessment><shipment_id>DEMO-001</shipment_id><image encoding="base64">' + base64.b64encode(photo.read_bytes()).decode() + '</image></assessment>'
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        response = client.post('/assessments', content=body, headers={'Content-Type':'application/xml'})
        assert response.status_code == 201, response.text
        parsed = ET.fromstring(response.content)
        value = float(parsed.findtext('load_pct'))
        assert 0 <= value <= 100
        assert ET.fromstring(client.get('/shipments/DEMO-001').content).findtext('assessment_id') == parsed.findtext('assessment_id')
        assert len(ET.fromstring(client.get('/shipments/DEMO-001/history').content)) == 1
        assert client.post('/assessments', content='<broken', headers={'Content-Type':'application/xml'}).status_code == 400
    with TestClient(create_app(db_path=db, predictor=predictor)) as client:
        assert client.get('/shipments/DEMO-001').status_code == 200
    out = root / 'docs/final_service_smoke.json'
    out.parent.mkdir(exist_ok=True)
    report = {'passed':True,'device':'cpu','prediction':value,'model_version':predictor.model_version,'processing_ms':float(parsed.findtext('processing_ms')),'startup_and_test_seconds':time.perf_counter()-start,'storage_survives_restart':True,'xml_error_checked':True,'exact_original_54545_weights':False}
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out.parent / 'final_service_example.xml').write_bytes(response.content)
    print(json.dumps(report,indent=2),flush=True)
