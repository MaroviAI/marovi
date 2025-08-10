from marovi.pipelines.core import PipelineStep, Pipeline, BranchingPipeline
from marovi.pipelines.context import PipelineContext


class MultiplyStep(PipelineStep[int, int]):
    def __init__(self, factor):
        super().__init__(step_id="mult")
        self.factor = factor

    def process(self, inputs, context):
        return [i * self.factor for i in inputs]


class AddStep(PipelineStep[int, int]):
    def __init__(self, add):
        super().__init__(step_id="add")
        self.add = add

    def process(self, inputs, context):
        return [i + self.add for i in inputs]


class FlakyStep(PipelineStep[int, int]):
    def __init__(self, fail_times=1):
        super().__init__(step_id="flaky")
        self.fail_times = fail_times

    def process(self, inputs, context):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ValueError("fail")
        return inputs


def test_run_with_retries():
    ctx = PipelineContext()
    step = FlakyStep(fail_times=1)
    assert step.run_with_retries([1], ctx) == [1]


def test_pipeline_execution(tmp_path):
    ctx = PipelineContext(checkpoint_dir=str(tmp_path))
    pipeline = Pipeline([
        MultiplyStep(2),
        AddStep(3),
    ], name="calc", checkpoint_dir=str(tmp_path))
    outputs = pipeline.run([1, 2], ctx)
    assert outputs == [5, 7]
    cp = tmp_path / f"{ctx.context_id}_calc_after_add.json"
    assert cp.exists()


def test_branching_pipeline(tmp_path):
    branches = {
        "main": [MultiplyStep(2)],
        "alt": [AddStep(1)],
    }
    pipeline = BranchingPipeline(branches, checkpoint_dir=str(tmp_path))
    ctx = PipelineContext(checkpoint_dir=str(tmp_path))
    assert pipeline.run([2], ctx) == [4]
    assert pipeline.run_branch("alt", [2], ctx) == [3]
