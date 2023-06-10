import pytest
import torch

from sheeprl.data.buffers import Trajectory, TrajectoryReplayBuffer


@pytest.fixture()
def trajectory():
    return Trajectory({"obs": torch.rand(5, 3), "rew": torch.rand(5, 1)}, batch_size=5)


@pytest.fixture()
def trajectory2():
    return Trajectory({"obs": torch.rand(7, 3), "rew": torch.rand(7, 1)}, batch_size=7)


def test_add_trajectory(trajectory):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=5)
    buffer.add(trajectory)
    assert len(buffer) == 1


def test_add_multiple_trajectories(trajectory, trajectory2):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=5)
    buffer.add(trajectory)
    buffer.add(trajectory2)
    assert len(buffer) == 2


def test_overflow_trajectories(trajectory, trajectory2):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=1)
    buffer.add(trajectory)
    buffer.add(trajectory2)
    assert len(buffer) == 1
    assert buffer[0].shape == torch.Size([7])


def test_sample_trajectory(trajectory, trajectory2):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=1)
    buffer.add(trajectory)
    buffer.add(trajectory2)
    sample = buffer.sample(batch_size=1, sequence_length=4)
    assert len(sample == 1)
    assert sample.shape == (1, 4)
    assert sample.valid_keys == ["obs", "rew"]
    assert sample["obs"].shape == (1, 4, 3)
    assert sample["rew"].shape == (1, 4, 1)


def test_sample_multiple_trajectories(trajectory, trajectory2):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=5)
    buffer.add(trajectory)
    buffer.add(trajectory2)
    buffer.add(trajectory)
    sample = buffer.sample(batch_size=2, sequence_length=4)
    assert len(sample == 2)
    assert sample.shape == (2, 4)
    assert sample.valid_keys == ["obs", "rew"]
    assert sample["obs"].shape == (2, 4, 3)
    assert sample["rew"].shape == (2, 4, 1)


def test_with_shorter_than_sequence_length_trajectory(trajectory):
    buffer = TrajectoryReplayBuffer(max_num_trajectories=5)
    short_traj = Trajectory({"obs": torch.rand(3, 3), "rew": torch.rand(3, 1)}, batch_size=3)
    buffer.add(short_traj)
    buffer.add(trajectory)
    sample = buffer.sample(batch_size=2, sequence_length=4)
    assert len(sample == 2)
    assert sample.shape == (2, 4)


def test_with_no_long_enough_trajectory():
    buffer = TrajectoryReplayBuffer(max_num_trajectories=5)
    short_traj = Trajectory({"obs": torch.rand(3, 3), "rew": torch.rand(3, 1)}, batch_size=3)
    buffer.add(short_traj)
    with pytest.raises(RuntimeError):
        sample = buffer.sample(batch_size=2, sequence_length=4)