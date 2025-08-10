from pathlib import Path
import json

from marovi.pipelines.context import PipelineContext


def test_context_state_and_checkpoint(tmp_path):
    ctx = PipelineContext(metadata={"doc": 1}, checkpoint_dir=str(tmp_path))
    ctx.log_step("step1", [1], [2])
    ctx.add_metadata("extra", "value")
    ctx.update_state("step1", [2], {"info": "x"})

    assert ctx.get_state("step1")["outputs"] == [2]
    assert ctx.get_outputs("step1") == [2]

    ctx.log_metrics({"loss": 0.5}, step=1)
    assert ctx.get_metric("loss") == 0.5

    ctx.register_artifact("model", "model.bin", {"size": 1})
    checkpoint = ctx.save_checkpoint("test")
    assert Path(checkpoint).exists()

    new_ctx = PipelineContext(checkpoint_dir=str(tmp_path))
    new_ctx.load_checkpoint(checkpoint)
    assert new_ctx.get_outputs("step1") == [2]

    json_str = ctx.to_json()
    restored = PipelineContext.from_json(json_str)
    assert restored.metadata["doc"] == 1

    summary = ctx.get_summary()
    assert summary["steps_executed"] == 1
