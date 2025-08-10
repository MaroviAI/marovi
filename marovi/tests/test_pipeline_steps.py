from marovi.pipelines.steps import ConditionalStep, CheckpointStep
from marovi.pipelines.context import PipelineContext
from marovi.pipelines.core import PipelineStep


class IdentityStep(PipelineStep[int, int]):
    def __init__(self, step_id="identity"):
        super().__init__(step_id=step_id)

    def process(self, inputs, context):
        return inputs


def test_conditional_step():
    ctx = PipelineContext()
    true_step = IdentityStep("true")
    false_step = IdentityStep("false")
    step = ConditionalStep(lambda x: x % 2 == 0, true_step, false_step, step_id="cond")
    outputs = step.process([1, 2, 3], ctx)
    # true branch processed even numbers first, then false branch
    assert outputs == [2, 1, 3]


def test_checkpoint_step(tmp_path):
    ctx = PipelineContext(checkpoint_dir=str(tmp_path))
    step = CheckpointStep(checkpoint_name="chk")
    data = [1, 2]
    outputs = step.process(data, ctx)
    assert outputs == data
    assert ctx.get_outputs("chk") == data
    checkpoint_path = tmp_path / f"{ctx.context_id}_chk.json"
    assert checkpoint_path.exists()
