"""LIBERO env whose object of interest is moved away from its spot after every reset.

In LIBERO-Object task 0 the soup can starts in nearly the same place in every layout, so a policy can
succeed by memorising where it is. Moving it a few centimetres tests whether the policy actually
looks at it.
"""
import numpy as np
from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action


class ShiftedLiberoEnv(LiberoEnv):
    """shift_cm: radius of the move. shift_uniform: draw the radius from U(0, shift_cm) instead.
    The direction (and radius) come from the reset seed, so a list of seeds is a fixed test set."""

    def __init__(self, *args, shift_cm=0.0, shift_uniform=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift_cm, self.shift_uniform = shift_cm, shift_uniform

    def reset(self, seed=None, **kwargs):
        obs, info = super().reset(seed=seed, **kwargs)
        info["shift"] = np.zeros(2)
        if self.shift_cm <= 0:
            return obs, info
        sim = self._env.env
        name = sim.obj_of_interest[0]
        joint = sim.objects_dict[name].joints[0]
        start = sim.sim.get_state()
        q0 = sim.sim.data.get_joint_qpos(joint).copy()
        rng = np.random.default_rng(seed)
        for _ in range(10):                 # retry a direction that knocks the object over
            r = self.shift_cm / 100 * (rng.uniform() if self.shift_uniform else 1.0)
            a = rng.uniform(0, 2 * np.pi)
            want = r * np.array([np.cos(a), np.sin(a)])
            sim.sim.set_state(start)
            q = q0.copy()
            q[:2] += want
            sim.sim.data.set_joint_qpos(joint, q)
            sim.sim.forward()
            for _ in range(self.num_steps_wait):
                raw, _, _, _ = self._env.step(get_libero_dummy_action())
            q1 = sim.sim.data.get_joint_qpos(joint)
            upright = abs(q1[2] - q0[2]) < 0.01
            if upright and np.linalg.norm(q1[:2] - q0[:2] - want) < 0.015:
                break
        else:
            raise RuntimeError(f"seed {seed}: could not place {name} {self.shift_cm} cm away")
        info["shift"] = q1[:2] - q0[:2]
        return self._format_raw_obs(raw), info

    def preview(self, plans, horizon):
        """Perfect foresight for one decision: run each candidate in plans[self.episode_index] ((K, >=horizon, 7))
        from the current state, then put the simulator back exactly where it was. Returns the observation
        after each candidate and whether the task was done on the way."""
        env = self._env
        sim, robot = env.env.sim, env.env.robots[0]
        state, warm = sim.get_state(), sim.data.qacc_warmstart.copy()
        grip, timestep = robot.gripper.current_action.copy(), env.env.timestep

        def restore():
            sim.set_state(state); sim.data.qacc_warmstart[:] = warm; sim.forward()
            robot.gripper.current_action = grip.copy(); env.env.timestep = timestep
            robot.controller.new_update = True

        obs, done = [], []
        for plan in plans[self.episode_index]:
            restore()
            d = False
            for act in plan[:horizon]:
                raw, _, _, _ = env.step(np.asarray(act, dtype=np.float64))
                if env.check_success():
                    d = True
                    break
            obs.append(self._format_raw_obs(raw)); done.append(d)
        restore()
        return obs, done

    def preview_checkpoints(self, plans, checkpoints):
        """Like preview(), but keeps the observation at every step count in `checkpoints` (sorted). An episode that
        finishes early keeps its last observation for the later checkpoints and counts as done there."""
        env = self._env
        sim, robot = env.env.sim, env.env.robots[0]
        state, warm = sim.get_state(), sim.data.qacc_warmstart.copy()
        grip, timestep = robot.gripper.current_action.copy(), env.env.timestep

        def restore():
            sim.set_state(state); sim.data.qacc_warmstart[:] = warm; sim.forward()
            robot.gripper.current_action = grip.copy(); env.env.timestep = timestep
            robot.controller.new_update = True

        obs, done = [], []
        for plan in plans[self.episode_index]:
            restore()
            d, raw, got_o, got_d = False, None, [], []
            for step in range(1, checkpoints[-1] + 1):
                if not d:
                    raw, _, _, _ = env.step(np.asarray(plan[step - 1], dtype=np.float64))
                    d = bool(env.check_success())
                if step in checkpoints:
                    got_o.append(self._format_raw_obs(raw)); got_d.append(d)
            obs.append(got_o); done.append(got_d)
        restore()
        return obs, done

    def snapshot(self):
        """Remember the simulator's exact state (see preview) so restore_snapshot can return to it."""
        env = self._env
        sim, robot = env.env.sim, env.env.robots[0]
        self._snap = (sim.get_state(), sim.data.qacc_warmstart.copy(), robot.gripper.current_action.copy(), env.env.timestep)

    def restore_snapshot(self):
        env = self._env
        sim, robot = env.env.sim, env.env.robots[0]
        state, warm, grip, timestep = self._snap
        sim.set_state(state); sim.data.qacc_warmstart[:] = warm; sim.forward()
        robot.gripper.current_action = grip.copy(); env.env.timestep = timestep
        robot.controller.new_update = True

    def raw_step(self, actions):
        """Step with actions[self.episode_index], bypassing the vector env's end-of-episode handling, so a
        branch can run past a success without freezing the sub-env. Returns (observation, task done)."""
        raw, _, _, _ = self._env.step(np.asarray(actions[self.episode_index], dtype=np.float64))
        return self._format_raw_obs(raw), bool(self._env.check_success())

    def can_position(self):
        sim = self._env.env
        return np.array(sim.sim.data.body_xpos[sim.obj_body_id[sim.obj_of_interest[0]]])

    def preview_probe(self, plans, horizon):
        """preview() that also returns the true position of the can now and after each candidate."""
        env = self._env
        sim, robot = env.env.sim, env.env.robots[0]
        state, warm = sim.get_state(), sim.data.qacc_warmstart.copy()
        grip, timestep = robot.gripper.current_action.copy(), env.env.timestep
        can_now = self.can_position()

        def restore():
            sim.set_state(state); sim.data.qacc_warmstart[:] = warm; sim.forward()
            robot.gripper.current_action = grip.copy(); env.env.timestep = timestep
            robot.controller.new_update = True

        obs, done, can = [], [], []
        for plan in plans[self.episode_index]:
            restore()
            d, raw = False, None
            for act in plan[:horizon]:
                raw, _, _, _ = env.step(np.asarray(act, dtype=np.float64))
                if env.check_success():
                    d = True
                    break
            obs.append(self._format_raw_obs(raw)); done.append(d); can.append(self.can_position())
        restore()
        return obs, done, can, can_now
