"""CPU-only structural gates; numerical/replay tests still require a GPU."""
import unittest
from exl3_ticket_source import ticket_source
from analyze_exl3_ticket_screen import schedule_model


class TicketTests(unittest.TestCase):
    source = '''    // Individual GEMM barriers per group
    locks += group_idx * MAX(hidden_dim, intermediate_dim) / 128;
    if (token_count == 0) continue;
    if (token_count > max_tokens_per_expert) continue;
    if (expert_idx_assign++ % concurrency != group_idx) continue;
    gemm_up(temp_state_g, temp_intermediate_g, exp_gate_trellis, K_gate);
    gemm_down(temp_intermediate_g, temp_state_g, exp_down_trellis, K_down);
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        };

        had_d_out();
    }
}'''

    def test_math_skips_and_barrier_count_preserved(self):
        result = ticket_source(self.source)
        for line in self.source.splitlines():
            if 'gemm_' in line or 'token_count' in line or 'locks +=' in line:
                self.assertIn(line, result)
        self.assertEqual(result.count('group_barrier('), self.source.count('group_barrier('))
        self.assertNotIn('% concurrency', result)

    def test_workspace_pointer_saved_before_lock_offset(self):
        result = ticket_source(self.source)
        self.assertLess(result.index('int* sched'), result.index('locks +='))
        self.assertIn('BARRIER_LOCKS_OFFSET + 1024', result)

    def test_acq_rel_retirement_and_both_resets(self):
        result = ticket_source(self.source)
        self.assertIn('retired.fetch_add(1, cuda::memory_order_acq_rel)', result)
        self.assertIn('next.store(0, cuda::memory_order_relaxed)', result)
        self.assertIn('retired.store(0, cuda::memory_order_relaxed)', result)
        self.assertLess(result.index('atomicAdd(sched'), result.index('group_barrier('))
        self.assertLess(result.index('group_barrier('), result.index('ticket = sched'))

    def test_drift_and_reapplication_rejected(self):
        for source in ['', self.source + self.source, ticket_source(self.source)]:
            with self.assertRaises(ValueError):
                ticket_source(source)

    def test_cost_model_equal_work_and_skips(self):
        r = schedule_model([0, 129] + [8]*36)
        self.assertEqual(r['round_robin_makespan'], 6)
        self.assertEqual(r['greedy_makespan'], 6)
        self.assertEqual(schedule_model([0, 129])['greedy_makespan'], 0)

    def test_cost_model_adversarial_work(self):
        r = schedule_model([48, 2, 2, 2, 2, 2]*6)
        self.assertEqual(r['round_robin_makespan'], 18)
        self.assertLess(r['greedy_makespan'], r['round_robin_makespan'])
        self.assertGreaterEqual(r['greedy_makespan'], r['optimistic_lower_bound'])


if __name__ == '__main__':
    unittest.main()
