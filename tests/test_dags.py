"""
Unit tests for Airflow DAGs — Day 7.

Tests verify DAG structure: task IDs, dependency order, schedule,
catchup setting, and max_active_runs. No Airflow database or scheduler
required — DAG objects are imported and inspected directly.

Soda gate logic and SparkSubmitOperator execution are not unit-tested
here (require live infrastructure). Covered by end-to-end tests on Day 9.
"""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("DELTA_BRONZE_PATH", "s3a://test/bronze")
os.environ.setdefault("DELTA_SILVER_PATH", "s3a://test/silver")
os.environ.setdefault("DELTA_GOLD_PATH",   "s3a://test/gold")
os.environ.setdefault("AWS_ACCESS_KEY_ID",     "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")


class TestFullPipelineDAG:
    def setup_method(self):
        from dags.full_pipeline_dag import dag
        self.dag = dag

    def test_dag_id(self):
        assert self.dag.dag_id == "rag_full_pipeline"

    def test_catchup_is_false(self):
        assert self.dag.catchup is False

    def test_max_active_runs_is_one(self):
        assert self.dag.max_active_runs == 1

    def test_schedule_is_every_30_minutes(self):
        # Note: Airflow 2.x stores schedule as dag.schedule_interval on the object.
        # When migrating to Airflow 3, change this assertion to dag.schedule.
        assert self.dag.schedule_interval == "*/30 * * * *"

    def test_expected_task_ids_present(self):
        task_ids = {t.task_id for t in self.dag.tasks}
        for expected in [
            "bronze_soda_gate",
            "silver_transform_job",
            "silver_soda_gate",
            "embedding_pipeline",
            "gold_soda_gate",
        ]:
            assert expected in task_ids

    def test_task_count(self):
        assert len(self.dag.tasks) == 5

    def test_bronze_gate_has_no_upstream(self):
        task = self.dag.get_task("bronze_soda_gate")
        assert len(task.upstream_task_ids) == 0

    def test_silver_job_downstream_of_bronze_gate(self):
        task = self.dag.get_task("silver_transform_job")
        assert "bronze_soda_gate" in task.upstream_task_ids

    def test_silver_gate_downstream_of_silver_job(self):
        task = self.dag.get_task("silver_soda_gate")
        assert "silver_transform_job" in task.upstream_task_ids

    def test_embedding_downstream_of_silver_gate(self):
        task = self.dag.get_task("embedding_pipeline")
        assert "silver_soda_gate" in task.upstream_task_ids

    def test_gold_gate_downstream_of_embedding(self):
        task = self.dag.get_task("gold_soda_gate")
        assert "embedding_pipeline" in task.upstream_task_ids

    def test_gold_gate_has_no_downstream(self):
        task = self.dag.get_task("gold_soda_gate")
        assert len(task.downstream_task_ids) == 0


class TestNightlyRefreshDAG:
    def setup_method(self):
        from dags.nightly_refresh_dag import dag
        self.dag = dag

    def test_dag_id(self):
        assert self.dag.dag_id == "rag_nightly_refresh"

    def test_catchup_is_false(self):
        assert self.dag.catchup is False

    def test_max_active_runs_is_one(self):
        assert self.dag.max_active_runs == 1

    def test_schedule_is_nightly_2am(self):
        # Note: Airflow 2.x stores schedule as dag.schedule_interval on the object.
        # When migrating to Airflow 3, change this assertion to dag.schedule.
        assert self.dag.schedule_interval == "0 2 * * *"

    def test_expected_task_ids_present(self):
        task_ids = {t.task_id for t in self.dag.tasks}
        for expected in ["freshness_check", "re_embedding_pipeline", "gold_soda_gate"]:
            assert expected in task_ids

    def test_task_count(self):
        assert len(self.dag.tasks) == 3

    def test_freshness_check_has_no_upstream(self):
        task = self.dag.get_task("freshness_check")
        assert len(task.upstream_task_ids) == 0

    def test_re_embedding_downstream_of_freshness_check(self):
        task = self.dag.get_task("re_embedding_pipeline")
        assert "freshness_check" in task.upstream_task_ids

    def test_gold_gate_downstream_of_re_embedding(self):
        task = self.dag.get_task("gold_soda_gate")
        assert "re_embedding_pipeline" in task.upstream_task_ids

    def test_gold_gate_is_leaf(self):
        task = self.dag.get_task("gold_soda_gate")
        assert len(task.downstream_task_ids) == 0
