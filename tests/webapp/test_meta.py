def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "tasks_loaded" in body


def test_tasks(client, stub_tasks):
    res = client.get("/api/meta/tasks")
    assert res.status_code == 200
    assert res.json()["tasks"] == stub_tasks


def test_registries(client):
    body = client.get("/api/meta/registries").json()
    keys = [m["key"] for m in body["model_types"]]
    assert "huggingface" in keys and "openai-chat-completions" in keys
    assert body["retrievers"] == ["pyserini"]
    assert body["rerankers"] == ["rerankers"]
    assert "bm25" in body["retriever_types"]


def test_presets_reshaped_from_gui_yaml(client):
    body = client.get("/api/meta/presets").json()
    # model types mirror gui_args.yaml's lm_eval_avil_model_types
    keys = {m["key"] for m in body["model_types"]}
    assert {"huggingface", "openai-chat-completions"} <= keys
    hf_presets = body["model_presets"]["huggingface"]
    assert all({"label", "args"} <= p.keys() for p in hf_presets)
    assert any("pretrained=" in p["args"] for p in hf_presets)
    assert "pyserini" in body["retriever_presets"]
    assert any("reranker_type=colbert" in s for s in body["reranker_presets"])


def test_devices(client):
    body = client.get("/api/meta/devices").json()
    assert "cpu" in body["devices"]


def test_no_frontend_message(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "not built" in res.text


def test_spa_serving_and_fallback(tmp_path, stub_tasks):
    from fastapi.testclient import TestClient

    from lrage.webapp.app import create_app
    from lrage.webapp.settings import Settings

    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html>spa</html>")
    (static / "assets" / "app.js").write_text("// js")

    app = create_app(
        Settings(data_dir=tmp_path / "data", static_dir=static),
        worker_fn=lambda ctx: None,
    )
    with TestClient(app) as client:
        assert client.get("/").text == "<html>spa</html>"
        # Client-side routes fall back to index.html...
        assert client.get("/runs/abc123/samples").text == "<html>spa</html>"
        # ...but real files and API routes are served as themselves.
        assert client.get("/assets/app.js").text == "// js"
        assert client.get("/api/health").status_code == 200
